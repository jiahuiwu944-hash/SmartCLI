from __future__ import annotations

import asyncio

import httpx

from paicli.agent.query import query
from paicli.config import PaiCliConfig
from paicli.context import ContextRuntime, TokenEstimator
from paicli.llm.errors import is_context_length_error
from paicli.tools import ToolRegistry
from paicli.types import Message


class SummaryClient:
    model_name = "test-model"
    provider_name = "test-provider"
    max_context_window = 16_384

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        yield {
            "type": "text_delta",
            "text": "- Goal preserved\n- write_file succeeded\n- tests passed",
        }
        yield {"type": "usage", "usage": {"input_tokens": 120, "output_tokens": 20}}
        yield {"type": "message_end", "stop_reason": "end_turn"}


def _tool_call(call_id: str, name: str = "read_file") -> Message:
    return Message(
        role="assistant",
        content="",
        tool_calls=[
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": '{"path":"large.py"}'},
            }
        ],
    )


def test_token_estimator_counts_system_messages_and_tools():
    estimator = TokenEstimator()
    base = estimator.estimate_request("system", [Message(role="user", content="hello")], [])
    expanded = estimator.estimate_request(
        "system prompt with more rules",
        [Message(role="user", content="hello 世界"), Message(role="assistant", content="answer")],
        [{"type": "function", "function": {"name": "read_file"}}],
    )

    assert base > 0
    assert expanded > base


def test_tool_result_slimming_preserves_atomic_tool_protocol():
    config = PaiCliConfig()
    config.memory.compression_threshold = 0.1
    config.memory.target_pressure_min = 0.99
    config.memory.target_pressure_max = 0.99
    config.memory.recent_turns_to_keep = 10
    runtime = ContextRuntime(SummaryClient(), config)
    current = Message(role="user", content="inspect the file")
    messages = [
        current,
        _tool_call("call-1"),
        Message(role="tool", tool_call_id="call-1", content="line\n" * 4000),
    ]

    prepared = asyncio.run(
        runtime.prepare(
            system_prompt="system",
            messages=messages,
            tools=[],
            protected_message=current,
        )
    )

    assistant = next(message for message in prepared.messages if message.tool_calls)
    result = next(message for message in prepared.messages if message.role == "tool")
    assert assistant.tool_calls[0]["id"] == result.tool_call_id
    assert "[SmartCLI compacted tool result]" in result.content
    assert prepared.compacted_tool_results == 1


def test_rolling_summary_preserves_current_request_and_recent_tool_pair():
    config = PaiCliConfig()
    config.memory.compression_threshold = 0.2
    config.memory.target_pressure_min = 0.5
    config.memory.target_pressure_max = 0.5
    config.memory.recent_turns_to_keep = 1
    runtime = ContextRuntime(SummaryClient(), config)
    current = Message(role="user", content="current task must survive")
    messages = [
        Message(role="user", content="old request " + "x" * 30_000),
        Message(role="assistant", content="old answer " + "y" * 20_000),
        current,
        _tool_call("recent-call", "execute_command"),
        Message(role="tool", tool_call_id="recent-call", content="tests passed"),
    ]

    prepared = asyncio.run(
        runtime.prepare(
            system_prompt="system",
            messages=messages,
            tools=[],
            protected_message=current,
        )
    )

    assert current in prepared.messages
    assert any(
        isinstance(message.content, str)
        and message.content.startswith("[SmartCLI conversation summary]")
        for message in prepared.messages
    )
    assistant = next(
        message
        for message in prepared.messages
        if any(call.get("id") == "recent-call" for call in message.tool_calls)
    )
    result = next(
        message for message in prepared.messages if message.tool_call_id == "recent-call"
    )
    assert assistant.tool_calls[0]["id"] == result.tool_call_id
    assert prepared.summarized_messages >= 2
    assert prepared.internal_input_tokens == 120
    assert prepared.internal_output_tokens == 20


def test_usage_calibration_is_smoothed_and_bounded():
    runtime = ContextRuntime(SummaryClient(), PaiCliConfig())

    runtime.observe_usage(actual_input_tokens=2000, estimated_input_tokens=1000)
    assert runtime.calibration_factor == 1.15

    for _ in range(20):
        runtime.observe_usage(actual_input_tokens=10_000, estimated_input_tokens=100)
    assert runtime.calibration_factor <= 1.5

    for _ in range(30):
        runtime.observe_usage(actual_input_tokens=1, estimated_input_tokens=1000)
    assert runtime.calibration_factor >= 0.7


def test_context_length_error_detection():
    request = httpx.Request("POST", "https://example.test/chat")
    response = httpx.Response(
        400,
        request=request,
        text='{"error":"maximum context length exceeded"}',
    )
    error = httpx.HTTPStatusError("bad request", request=request, response=response)

    assert is_context_length_error(error) is True
    assert is_context_length_error(httpx.ConnectError("offline", request=request)) is False


class ContextRetryClient(SummaryClient):
    def __init__(self):
        self.calls = 0

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        self.calls += 1
        if self.calls == 1:
            request = httpx.Request("POST", "https://example.test/chat")
            response = httpx.Response(
                400,
                request=request,
                text='{"error":"maximum context length exceeded"}',
            )
            yield {
                "type": "error",
                "error": httpx.HTTPStatusError(
                    "bad request", request=request, response=response
                ),
            }
            return
        yield {"type": "text_delta", "text": "recovered"}
        yield {"type": "message_end", "stop_reason": "end_turn"}


def test_query_emergency_compresses_and_retries_once(tmp_path):
    client = ContextRetryClient()
    config = PaiCliConfig()
    config.agent.stop_hook_enabled = False

    async def collect():
        return [
            event
            async for event in query(
                llm_client=client,
                tool_registry=ToolRegistry(),
                system_prompt="system",
                user_message="finish the task",
                history=[],
                cwd=str(tmp_path),
                config=config,
            )
        ]

    events = asyncio.run(collect())

    assert client.calls == 2
    assert any(
        event["type"] == "context_compressed" and event["emergency"]
        for event in events
    )
    done = next(event for event in events if event["type"] == "done")
    assert done["completed"] is True
    assert done["total_turns"] == 1
