from __future__ import annotations

import asyncio
import importlib
from typing import Any

import httpx

from paicli.agent import Agent, QueryEngine
from paicli.agent.stop_hook import verify_answer
from paicli.config import load_config
from paicli.tools import ToolRegistry, get_builtin_tools
from paicli.types import Message


class FakeClient:
    model_name = "fake-model"
    provider_name = "fake-provider"
    max_context_window = 1000

    def __init__(self):
        self.calls = 0

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        if "Stop Hook reviewer" in system_prompt:
            yield {"type": "text_delta", "text": '{"approved": true, "feedback": ""}'}
            yield {"type": "message_end", "stop_reason": "end_turn"}
            return
        self.calls += 1
        if self.calls == 1:
            yield {
                "type": "tool_call_delta",
                "tool_call": {
                    "index": 0,
                    "id": "call_1",
                    "function": {"name": "read_file", "arguments": '{"path":"note.txt"}'},
                },
            }
            yield {"type": "message_end", "stop_reason": "tool_use"}
        else:
            tool_messages = [message for message in messages if message.role == "tool"]
            assert tool_messages
            assert "1: hello" in tool_messages[-1].content
            yield {"type": "text_delta", "text": "done"}
            yield {"type": "message_end", "stop_reason": "end_turn"}


def test_query_engine_executes_tool_and_replays_result(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "note.txt").write_text("hello\n", encoding="utf-8")
    config = load_config(project_root=tmp_path)
    registry = ToolRegistry()
    registry.register_all(get_builtin_tools())
    engine = QueryEngine(
        llm_client=FakeClient(),
        tool_registry=registry,
        config=config,
        cwd=str(tmp_path),
    )

    async def run() -> Any:
        return await engine.ask_complete_async("read note")

    result = asyncio.run(run())
    assert result.text == "done"
    assert result.turns == 2
    assert result.completed is True
    assert result.termination_reason == "completed"


class ToolFailureObservationClient:
    model_name = "fake-model"
    provider_name = "fake-provider"
    max_context_window = 1000

    def __init__(self):
        self.calls = 0

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        if "Stop Hook reviewer" in system_prompt:
            yield {"type": "text_delta", "text": '{"approved": true, "feedback": ""}'}
            yield {"type": "message_end", "stop_reason": "end_turn"}
            return
        self.calls += 1
        if self.calls == 1:
            yield {
                "type": "tool_call_delta",
                "tool_call": {
                    "index": 0,
                    "id": "missing_call",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path":"missing.txt"}',
                    },
                },
            }
            yield {"type": "message_end", "stop_reason": "tool_use"}
            return

        tool_messages = [message for message in messages if message.role == "tool"]
        assert tool_messages
        assert 'Tool "read_file" execution error' in tool_messages[-1].content
        yield {"type": "text_delta", "text": "changed approach after observing the failure"}
        yield {"type": "message_end", "stop_reason": "end_turn"}


def test_tool_business_failure_is_returned_to_next_react_turn(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config = load_config(project_root=tmp_path)
    registry = ToolRegistry()
    registry.register_all(get_builtin_tools())
    engine = QueryEngine(
        llm_client=ToolFailureObservationClient(),
        tool_registry=registry,
        config=config,
        cwd=str(tmp_path),
    )

    events = asyncio.run(_collect_events(engine))

    result = next(event for event in events if event["type"] == "tool_result")
    done = next(event for event in events if event["type"] == "done")
    assert result["is_error"] is True
    assert done["completed"] is True


class LoopingToolClient:
    model_name = "fake-model"
    provider_name = "fake-provider"
    max_context_window = 1000

    def __init__(self, *, repeated: bool, include_usage: bool = False):
        self.calls = 0
        self.repeated = repeated
        self.include_usage = include_usage

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        if "Stop Hook reviewer" in system_prompt:
            yield {"type": "text_delta", "text": '{"approved": true, "feedback": ""}'}
            yield {"type": "message_end", "stop_reason": "end_turn"}
            return
        self.calls += 1
        path = "note.txt" if self.repeated else f"missing-{self.calls}.txt"
        yield {
            "type": "tool_call_delta",
            "tool_call": {
                "index": 0,
                "id": f"call_{self.calls}",
                "function": {"name": "read_file", "arguments": f'{{"path":"{path}"}}'},
            },
        }
        if self.include_usage:
            yield {"type": "usage", "usage": {"input_tokens": 80, "output_tokens": 20}}
        yield {"type": "message_end", "stop_reason": "tool_use"}


def _engine(tmp_path, config, client, **kwargs):
    registry = ToolRegistry()
    registry.register_all(get_builtin_tools())
    return QueryEngine(
        llm_client=client,
        tool_registry=registry,
        config=config,
        cwd=str(tmp_path),
        **kwargs,
    )


async def _collect_events(engine: QueryEngine) -> list[dict[str, Any]]:
    return [event async for event in engine.ask("keep working")]


def test_token_budget_stops_before_pending_tool_execution(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "note.txt").write_text("hello\n", encoding="utf-8")
    config = load_config(project_root=tmp_path)
    config.agent.max_total_tokens = 100
    client = LoopingToolClient(repeated=True, include_usage=True)

    events = asyncio.run(_collect_events(_engine(tmp_path, config, client)))

    stopped = next(event for event in events if event["type"] == "run_stopped")
    done = next(event for event in events if event["type"] == "done")
    assert stopped["reason"] == "token_budget"
    assert not any(event["type"] == "tool_result" for event in events)
    assert done["completed"] is False
    assert done["termination_reason"] == "token_budget"


def test_repeated_identical_tool_call_redirects_model_instead_of_stopping(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "note.txt").write_text("hello\n", encoding="utf-8")
    config = load_config(project_root=tmp_path)
    config.agent.repeated_tool_call_limit = 3
    config.agent.max_total_tokens = 0
    client = AdaptiveRepeatClient()

    events = asyncio.run(_collect_events(_engine(tmp_path, config, client)))

    redirected = next(
        event
        for event in events
        if event["type"] == "model_redirected" and event["reason"] == "repeated_tool_call"
    )
    done = next(event for event in events if event["type"] == "done")
    assert "change the parameters" in redirected["message"]
    assert len([event for event in events if event["type"] == "tool_result"]) == 2
    assert done["completed"] is True
    assert done["termination_reason"] == "completed"


class RecoveringToolErrorClient:
    model_name = "fake-model"
    provider_name = "fake-provider"
    max_context_window = 1000

    def __init__(self):
        self.calls = 0

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        if "Stop Hook reviewer" in system_prompt:
            yield {"type": "text_delta", "text": '{"approved": true, "feedback": ""}'}
            yield {"type": "message_end", "stop_reason": "end_turn"}
            return

        self.calls += 1
        if self.calls <= 2:
            yield {
                "type": "tool_call_delta",
                "tool_call": {
                    "index": 0,
                    "id": f"call_{self.calls}",
                    "function": {
                        "name": "read_file",
                        "arguments": f'{{"path":"missing-{self.calls}.txt"}}',
                    },
                },
            }
            yield {"type": "message_end", "stop_reason": "tool_use"}
            return

        assert any(
            message.role == "user"
            and "All tool calls failed for 2 consecutive turns" in str(message.content)
            for message in messages
        )
        yield {"type": "text_delta", "text": "The requested resource is unavailable."}
        yield {"type": "message_end", "stop_reason": "end_turn"}


def test_consecutive_tool_error_turns_redirect_the_model(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config = load_config(project_root=tmp_path)
    config.agent.consecutive_tool_error_limit = 2
    config.agent.max_total_tokens = 0
    client = RecoveringToolErrorClient()

    events = asyncio.run(_collect_events(_engine(tmp_path, config, client)))

    redirected = next(
        event
        for event in events
        if event["type"] == "model_redirected" and event["reason"] == "consecutive_tool_errors"
    )
    results = [event for event in events if event["type"] == "tool_result"]
    done = next(event for event in events if event["type"] == "done")
    assert "change the command" in redirected["message"]
    assert len(results) == 2
    assert all(event["is_error"] for event in results)
    assert not any(event["type"] == "run_stopped" for event in events)
    assert done["completed"] is True


def test_max_turns_remains_a_last_resort_safety_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config = load_config(project_root=tmp_path)
    config.agent.max_turns = 2
    config.agent.max_total_tokens = 0
    config.agent.consecutive_tool_error_limit = 0
    client = LoopingToolClient(repeated=False)

    events = asyncio.run(_collect_events(_engine(tmp_path, config, client)))

    stopped = next(event for event in events if event["type"] == "run_stopped")
    requested = next(event for event in events if event["type"] == "budget_extension_requested")
    done = next(event for event in events if event["type"] == "done")
    assert stopped["reason"] == "max_turns"
    assert requested["context_preserved"] is True
    assert done["total_turns"] == 2


def test_runtime_budget_stops_before_pending_tool_execution(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "note.txt").write_text("hello\n", encoding="utf-8")
    config = load_config(project_root=tmp_path)
    config.agent.max_runtime_seconds = 1
    config.agent.max_total_tokens = 0
    client = LoopingToolClient(repeated=True)
    query_module = importlib.import_module("paicli.agent.query")
    ticks = iter([0.0, 0.1, 1.1])
    monkeypatch.setattr(query_module, "monotonic", lambda: next(ticks))

    events = asyncio.run(_collect_events(_engine(tmp_path, config, client)))

    stopped = next(event for event in events if event["type"] == "run_stopped")
    assert stopped["reason"] == "runtime_budget"
    assert not any(event["type"] == "tool_result" for event in events)


class AdaptiveRepeatClient:
    model_name = "fake-model"
    provider_name = "fake-provider"
    max_context_window = 1000

    def __init__(self):
        self.calls = 0

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        if "Stop Hook reviewer" in system_prompt:
            yield {"type": "text_delta", "text": '{"approved": true, "feedback": ""}'}
            yield {"type": "message_end", "stop_reason": "end_turn"}
            return
        self.calls += 1
        if messages[-1].role == "tool" and "change the parameters" in messages[-1].content:
            yield {"type": "text_delta", "text": "changed approach and finished"}
            yield {"type": "message_end", "stop_reason": "end_turn"}
            return
        yield {
            "type": "tool_call_delta",
            "tool_call": {
                "index": 0,
                "id": f"repeat_{self.calls}",
                "function": {"name": "read_file", "arguments": '{"path":"note.txt"}'},
            },
        }
        yield {"type": "message_end", "stop_reason": "tool_use"}


class UsageToolThenFinalClient(FakeClient):
    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        async for event in super().chat(messages, tools, system_prompt=system_prompt):
            yield event
            if event.get("type") == "message_end":
                yield {"type": "usage", "usage": {"input_tokens": 80, "output_tokens": 20}}


class RevisingStopHookClient:
    model_name = "fake-model"
    provider_name = "fake-provider"
    max_context_window = 1000

    def __init__(self):
        self.main_calls = 0
        self.review_calls = 0

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        if "Stop Hook reviewer" in system_prompt:
            self.review_calls += 1
            verdict = (
                '{"approved": false, "feedback": "Run the tests before stopping."}'
                if self.review_calls == 1
                else '{"approved": true, "feedback": "Tests verified."}'
            )
            yield {"type": "text_delta", "text": verdict}
            yield {"type": "message_end", "stop_reason": "end_turn"}
            return
        self.main_calls += 1
        text = "draft answer" if self.main_calls == 1 else "revised answer with test evidence"
        yield {"type": "text_delta", "text": text}
        yield {"type": "message_end", "stop_reason": "end_turn"}


def test_max_turn_budget_can_be_extended_and_context_continues(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "note.txt").write_text("hello\n", encoding="utf-8")
    config = load_config(project_root=tmp_path)
    config.agent.max_turns = 1
    config.agent.stop_hook_enabled = False
    requests = []

    def approve(request):
        requests.append(request)
        return True

    engine = _engine(
        tmp_path,
        config,
        FakeClient(),
        continuation_callback=approve,
    )
    events = asyncio.run(_collect_events(engine))

    extended = next(event for event in events if event["type"] == "budget_extended")
    done = next(event for event in events if event["type"] == "done")
    assert requests[0]["reason"] == "max_turns"
    assert extended["context_preserved"] is True
    assert extended["additional_turns"] == config.agent.budget_extension_turns
    assert done["completed"] is True
    assert done["total_turns"] == 2
    assert any(message.role == "tool" for message in done["messages"])


def test_token_budget_can_be_extended_before_pending_tool_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "note.txt").write_text("hello\n", encoding="utf-8")
    config = load_config(project_root=tmp_path)
    config.agent.max_total_tokens = 100
    config.agent.stop_hook_enabled = False
    engine = _engine(
        tmp_path,
        config,
        UsageToolThenFinalClient(),
        continuation_callback=lambda request: {"continue": True},
    )

    events = asyncio.run(_collect_events(engine))

    extended = next(event for event in events if event["type"] == "budget_extended")
    assert extended["reason"] == "token_budget"
    assert extended["additional_tokens"] == config.agent.budget_extension_tokens
    assert any(event["type"] == "tool_result" for event in events)
    assert next(event for event in events if event["type"] == "done")["completed"] is True


def test_stop_hook_rejects_draft_and_agent_revises_before_stopping(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config = load_config(project_root=tmp_path)
    config.agent.max_total_tokens = 0
    client = RevisingStopHookClient()

    events = asyncio.run(_collect_events(_engine(tmp_path, config, client)))

    reviews = [event for event in events if event["type"] == "stop_hook_review"]
    redirects = [event for event in events if event["type"] == "model_redirected"]
    done = next(event for event in events if event["type"] == "done")
    assert [event["approved"] for event in reviews] == [False, True]
    assert redirects[0]["reason"] == "stop_hook_rejected"
    assert client.main_calls == 2
    assert client.review_calls == 2
    assert done["completed"] is True


class UnexpectedReviewClient:
    model_name = "fake-model"
    provider_name = "fake-provider"
    max_context_window = 1000

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        raise AssertionError("deterministic contradiction should reject before an LLM review")
        yield  # pragma: no cover


def test_stop_hook_deterministically_rejects_blocked_but_completed_claim():
    messages = _blocked_tool_messages()

    result = asyncio.run(
        verify_answer(
            llm_client=UnexpectedReviewClient(),
            original_request="连续读取三次",
            proposed_answer="三次读取已完成，但第三次被拦截并未执行。",
            messages=messages,
        )
    )

    assert result.approved is False
    assert "not counted as completed" in result.feedback


class ApprovingReviewClient:
    model_name = "fake-model"
    provider_name = "fake-provider"
    max_context_window = 1000

    def __init__(self):
        self.payload = ""

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        self.payload = str(messages[0].content)
        yield {"type": "text_delta", "text": '{"approved": true, "feedback": ""}'}
        yield {"type": "message_end", "stop_reason": "end_turn"}


def test_stop_hook_allows_honest_partial_completion_after_a_block():
    result = asyncio.run(
        verify_answer(
            llm_client=ApprovingReviewClient(),
            original_request="连续读取三次",
            proposed_answer="原始任务未完成：前两次已完成，第三次被拦截且没有执行。",
            messages=_blocked_tool_messages(),
        )
    )

    assert result.approved is True


def test_stop_hook_does_not_treat_words_inside_source_as_a_runtime_guard():
    messages = [
        Message(
            role="assistant",
            content="",
            tool_calls=[
                {
                    "id": "source_read",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path":"stop_hook.py"}',
                    },
                }
            ],
        ),
        Message(
            role="tool",
            tool_call_id="source_read",
            content=(
                "[FILE_CONTENT]\n"
                'markers = ("tool call was blocked", "calls were not executed again")'
            ),
        ),
    ]

    result = asyncio.run(
        verify_answer(
            llm_client=ApprovingReviewClient(),
            original_request="Explain the source code.",
            proposed_answer="The source defines guard-message markers.",
            messages=messages,
        )
    )

    assert result.approved is True


def test_stop_hook_evidence_summarizes_successful_read_file_coverage():
    client = ApprovingReviewClient()
    messages = [
        Message(
            role="assistant",
            content="",
            tool_calls=[
                {
                    "id": "read_range",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path":"README.md","offset":101,"limit":2}',
                    },
                }
            ],
        ),
        Message(
            role="tool",
            tool_call_id="read_range",
            content=(
                "[FILE_METADATA]\n"
                "path: README.md\n"
                "version: sha256:abc\n"
                "size: 5000\n"
                "[FILE_CONTENT]\n"
                "101: first line\n"
                "102: second line\n"
            ),
        ),
    ]

    result = asyncio.run(
        verify_answer(
            llm_client=client,
            original_request="Inspect these lines.",
            proposed_answer="The requested range was inspected.",
            messages=messages,
        )
    )

    assert result.approved is True
    assert "read_succeeded path=README.md" in client.payload
    assert "returned_lines=101-102" in client.payload
    assert "version=sha256:abc" in client.payload


class AlwaysRejectingStopHookClient:
    model_name = "fake-model"
    provider_name = "fake-provider"
    max_context_window = 1000

    def __init__(self):
        self.main_calls = 0

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        if "Stop Hook reviewer" in system_prompt:
            yield {
                "type": "text_delta",
                "text": '{"approved": false, "feedback": "Need more evidence."}',
            }
            yield {"type": "message_end", "stop_reason": "end_turn"}
            return
        self.main_calls += 1
        yield {"type": "text_delta", "text": f"candidate answer {self.main_calls}"}
        yield {"type": "message_end", "stop_reason": "end_turn"}


def test_stop_hook_exhaustion_does_not_promote_rejected_draft(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config = load_config(project_root=tmp_path)
    config.agent.max_total_tokens = 0
    config.agent.stop_hook_max_retries = 1

    events = asyncio.run(
        _collect_events(_engine(tmp_path, config, AlwaysRejectingStopHookClient()))
    )

    stopped = next(event for event in events if event["type"] == "run_stopped")
    done = next(event for event in events if event["type"] == "done")
    assert not any(event["type"] == "answer_preserved" for event in events)
    assert stopped["reason"] == "stop_hook_retries_exhausted"
    assert "No unverified draft was shown" in stopped["message"]
    assert done["completed"] is False


def _blocked_tool_messages() -> list[Message]:
    return [
        Message(
            role="assistant",
            content="",
            tool_calls=[
                {
                    "id": "call_3",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path":"README.md"}',
                    },
                }
            ],
        ),
        Message(
            role="tool",
            tool_call_id="call_3",
            content=(
                "SmartCLI safety guard: The identical call was blocked and the call "
                "was not executed again."
            ),
        ),
    ]


class RecoveringConnectionClient:
    model_name = "fake-model"
    provider_name = "fake-provider"
    max_context_window = 1000

    def __init__(self):
        self.calls = 0
        self.seen_messages = []

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        self.calls += 1
        self.seen_messages.append(list(messages))
        if self.calls == 1:
            raise httpx.ConnectError("service offline")
        yield {"type": "text_delta", "text": "continued successfully"}
        yield {"type": "message_end", "stop_reason": "end_turn"}


def test_connection_error_is_friendly_and_agent_context_can_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config = load_config(project_root=tmp_path)
    config.agent.stop_hook_enabled = False
    client = RecoveringConnectionClient()
    registry = ToolRegistry()
    registry.register_all(get_builtin_tools())
    agent = Agent(
        llm_client=client,
        tool_registry=registry,
        system_prompt="test",
        cwd=str(tmp_path),
        config=config,
    )

    async def run(message):
        return [event async for event in agent.run(message)]

    first_events = asyncio.run(run("inspect the project"))
    error = next(event for event in first_events if event["type"] == "error")
    assert error["recoverable"] is True
    assert error["context_preserved"] is True
    assert "当前上下文已保留" in error["message"]
    assert any(
        message.role == "user" and message.content == "inspect the project"
        for message in agent.history
    )

    second_events = asyncio.run(run("继续"))
    assert next(event for event in second_events if event["type"] == "done")["completed"]
    second_context = client.seen_messages[1]
    assert [message.content for message in second_context if message.role == "user"] == [
        "inspect the project",
        "继续",
    ]
