from __future__ import annotations

import json
from collections.abc import AsyncIterator
from inspect import isawaitable
from time import monotonic
from typing import Any

from paicli.agent.stop_hook import StopHookResult, verify_answer
from paicli.config import PaiCliConfig
from paicli.image import parse_image_references
from paicli.llm.base import LlmClient
from paicli.llm.errors import friendly_llm_error, llm_error_event
from paicli.tools.base import ToolContext
from paicli.tools.executor import ToolExecutor
from paicli.tools.registry import ToolRegistry
from paicli.types import Message


async def query(
    *,
    llm_client: LlmClient,
    tool_registry: ToolRegistry,
    system_prompt: str,
    user_message: str,
    history: list[Message] | None,
    cwd: str,
    config: PaiCliConfig,
    approval_callback=None,
    continuation_callback=None,
    stop_hook_callback=None,
    skill_context_buffer=None,
    max_turns: int | None = None,
    stop_hook_enabled: bool | None = None,
) -> AsyncIterator[dict[str, Any]]:
    original_request = user_message
    user_message = _prepend_skill_context(user_message, skill_context_buffer)
    messages = [
        *(history or []),
        Message(role="user", content=parse_image_references(user_message, cwd)),
    ]
    tool_definitions = tool_registry.definitions()
    executor = ToolExecutor(tool_registry)
    context = ToolContext(
        cwd=cwd,
        config=config,
        approval_callback=approval_callback,
        skill_context_buffer=skill_context_buffer,
    )

    total_tokens = 0
    turn = 0
    started_at = monotonic()
    turn_limit = _positive_int(
        max_turns if max_turns is not None else config.agent.max_turns,
        default=20,
    )
    token_budget = _optional_positive_int(config.agent.max_total_tokens)
    runtime_budget = _optional_positive_float(config.agent.max_runtime_seconds)
    repeat_limit = _optional_positive_int(config.agent.repeated_tool_call_limit)
    error_limit = _optional_positive_int(config.agent.consecutive_tool_error_limit)
    use_stop_hook = (
        config.agent.stop_hook_enabled if stop_hook_enabled is None else stop_hook_enabled
    )
    stop_hook_retry_limit = _optional_positive_int(config.agent.stop_hook_max_retries)
    extension_turns = _positive_int(config.agent.budget_extension_turns, default=20)
    extension_tokens = _positive_int(
        config.agent.budget_extension_tokens,
        default=max(token_budget, 100_000),
    )
    last_tool_signature = ""
    identical_tool_streak = 0
    consecutive_tool_error_turns = 0
    stop_hook_retries = 0
    termination_reason = "completed"
    termination_message = ""

    while True:
        reached_budgets = _reached_budgets(
            turn=turn,
            turn_limit=turn_limit,
            total_tokens=total_tokens,
            token_budget=token_budget,
        )
        if reached_budgets:
            request = _budget_extension_request(
                reached_budgets,
                turn=turn,
                turn_limit=turn_limit,
                total_tokens=total_tokens,
                token_budget=token_budget,
                extension_turns=extension_turns,
                extension_tokens=extension_tokens,
            )
            yield {"type": "budget_extension_requested", **request}
            decision = await _resolve_budget_extension(continuation_callback, request)
            if decision:
                added_turns = decision.get("additional_turns", 0)
                added_tokens = decision.get("additional_tokens", 0)
                turn_limit += added_turns
                if added_tokens:
                    token_budget = (token_budget or total_tokens) + added_tokens
                yield {
                    "type": "budget_extended",
                    "reason": "+".join(reached_budgets),
                    "additional_turns": added_turns,
                    "additional_tokens": added_tokens,
                    "turn_limit": turn_limit,
                    "token_budget": token_budget,
                    "context_preserved": True,
                }
            else:
                termination_reason = "+".join(reached_budgets)
                termination_message = (
                    "Execution budget was reached and no additional budget was approved. "
                    "The full conversation context has been preserved for a later continuation."
                )
                yield _run_stopped_event(
                    termination_reason,
                    termination_message,
                    turn,
                    total_tokens,
                )
                break
        if runtime_budget and monotonic() - started_at >= runtime_budget:
            termination_reason = "runtime_budget"
            termination_message = (
                f"Runtime budget reached ({runtime_budget:g}s). "
                "The run stopped before starting another model turn."
            )
            yield _run_stopped_event(
                termination_reason,
                termination_message,
                turn,
                total_tokens,
            )
            break

        turn += 1
        text = ""
        thinking = ""
        stop_reason = "end_turn"
        usage_input = 0
        usage_output = 0
        tool_states: dict[int, dict[str, Any]] = {}
        stream_error: Any = None

        try:
            async for event in llm_client.chat(
                messages,
                tool_definitions,
                system_prompt=system_prompt,
            ):
                event_type = event.get("type")
                if event_type == "text_delta":
                    delta = str(event.get("text") or "")
                    text += delta
                    yield {"type": "text_delta", "text": delta}
                elif event_type == "thinking_delta":
                    delta = str(event.get("thinking") or "")
                    thinking += delta
                    yield {"type": "thinking_delta", "thinking": delta}
                elif event_type == "tool_call_delta":
                    _merge_tool_delta(tool_states, event["tool_call"])
                elif event_type == "message_end":
                    stop_reason = str(event.get("stop_reason") or "end_turn")
                elif event_type == "usage":
                    usage = event.get("usage") or {}
                    usage_input += int(usage.get("input_tokens") or 0)
                    usage_output += int(usage.get("output_tokens") or 0)
                    yield {"type": "usage", "usage": usage}
                elif event_type == "error":
                    stream_error = event.get("error") or "Unknown model stream error"
                    break
        except Exception as exc:  # noqa: BLE001
            stream_error = exc

        if stream_error is not None:
            total_tokens += usage_input + usage_output
            if text:
                messages.append(
                    Message(
                        role="assistant",
                        content=text + "\n\n[Response interrupted by a model connection error.]",
                    )
                )
            termination_reason = "llm_connection_error"
            termination_message = friendly_llm_error(stream_error)
            yield llm_error_event(stream_error, messages=messages)
            yield {
                "type": "done",
                "total_turns": turn,
                "total_tokens": total_tokens,
                "termination_reason": termination_reason,
                "termination_message": termination_message,
                "completed": False,
                "messages": messages,
            }
            return

        total_tokens += usage_input + usage_output
        tool_calls = _finalize_tool_calls(tool_states)
        assistant_message = Message(role="assistant", content=text, tool_calls=tool_calls)
        if thinking and text:
            assistant_message.content = text
        elif thinking:
            assistant_message.content = ""
        messages.append(assistant_message)
        yield {"type": "turn_complete", "turn": turn, "stop_reason": stop_reason}

        if stop_reason != "tool_use" and not tool_calls:
            if stop_reason == "max_tokens":
                feedback = (
                    "Your previous response was truncated by the model output-token limit. "
                    "Continue from where it stopped and finish the task without repeating it."
                )
                messages.append(Message(role="user", content=f"[Runtime feedback]\n{feedback}"))
                yield {
                    "type": "model_redirected",
                    "reason": "model_token_limit",
                    "message": feedback,
                }
                continue

            if not use_stop_hook:
                break
            try:
                hook_result = await _run_stop_hook(
                    stop_hook_callback,
                    llm_client=llm_client,
                    original_request=original_request,
                    proposed_answer=text,
                    messages=messages,
                )
            except Exception as exc:  # noqa: BLE001
                termination_reason = "stop_hook_connection_error"
                termination_message = friendly_llm_error(exc, source="stop_hook")
                yield llm_error_event(exc, messages=messages, source="stop_hook")
                break
            total_tokens += hook_result.input_tokens + hook_result.output_tokens
            if hook_result.input_tokens or hook_result.output_tokens:
                yield {
                    "type": "usage",
                    "usage": {
                        "input_tokens": hook_result.input_tokens,
                        "output_tokens": hook_result.output_tokens,
                    },
                    "source": "stop_hook",
                }
            yield {
                "type": "stop_hook_review",
                "approved": hook_result.approved,
                "feedback": hook_result.feedback,
                "attempt": stop_hook_retries + 1,
            }
            if hook_result.approved:
                break

            stop_hook_retries += 1
            if stop_hook_retry_limit and stop_hook_retries > stop_hook_retry_limit:
                termination_reason = "stop_hook_retries_exhausted"
                termination_message = (
                    "Stop Hook still found the result incomplete after "
                    f"{stop_hook_retry_limit} correction attempts: {hook_result.feedback}"
                )
                yield _run_stopped_event(
                    termination_reason,
                    termination_message,
                    turn,
                    total_tokens,
                )
                break
            messages.append(
                Message(
                    role="user",
                    content=(
                        "[Stop Hook rejected the attempted final answer]\n"
                        f"{hook_result.feedback}\n"
                        "Do not stop yet. Use tools if needed, correct the work, verify it, "
                        "and then provide a revised final answer."
                    ),
                )
            )
            yield {
                "type": "model_redirected",
                "reason": "stop_hook_rejected",
                "message": hook_result.feedback,
            }
            continue

        if not tool_calls:
            termination_reason = "invalid_tool_request"
            termination_message = (
                "The model requested tool use but did not produce a valid tool call."
            )
            yield _run_stopped_event(
                termination_reason,
                termination_message,
                turn,
                total_tokens,
            )
            break

        if token_budget and total_tokens >= token_budget:
            request = _budget_extension_request(
                ["token_budget"],
                turn=turn,
                turn_limit=turn_limit,
                total_tokens=total_tokens,
                token_budget=token_budget,
                extension_turns=extension_turns,
                extension_tokens=extension_tokens,
            )
            yield {"type": "budget_extension_requested", **request}
            decision = await _resolve_budget_extension(continuation_callback, request)
            if decision:
                added_tokens = decision.get("additional_tokens", 0)
                token_budget = (token_budget or total_tokens) + added_tokens
                yield {
                    "type": "budget_extended",
                    "reason": "token_budget",
                    "additional_turns": 0,
                    "additional_tokens": added_tokens,
                    "turn_limit": turn_limit,
                    "token_budget": token_budget,
                    "context_preserved": True,
                }
            else:
                termination_reason = "token_budget"
                termination_message = (
                    f"Run token budget reached ({total_tokens}/{token_budget}); additional "
                    "budget was not approved. Context was preserved."
                )
                _append_guard_tool_results(messages, tool_calls, termination_message)
                yield _run_stopped_event(
                    termination_reason,
                    termination_message,
                    turn,
                    total_tokens,
                )
                break

        if runtime_budget and monotonic() - started_at >= runtime_budget:
            termination_reason = "runtime_budget"
            termination_message = (
                f"Runtime budget reached ({runtime_budget:g}s). "
                "Pending tool calls were not executed."
            )
            _append_guard_tool_results(messages, tool_calls, termination_message)
            yield _run_stopped_event(
                termination_reason,
                termination_message,
                turn,
                total_tokens,
            )
            break

        tool_signature = _tool_batch_signature(tool_calls)
        if tool_signature == last_tool_signature:
            identical_tool_streak += 1
        else:
            last_tool_signature = tool_signature
            identical_tool_streak = 1
        if repeat_limit and identical_tool_streak >= repeat_limit:
            repeated_names = ", ".join(
                str(call.get("function", {}).get("name") or "unknown") for call in tool_calls
            )
            correction = (
                f"The identical tool call and parameters were requested "
                f"{identical_tool_streak} consecutive times ({repeated_names}). The calls were "
                "not executed again. Reassess the latest result, change the parameters or use a "
                "different tool/approach; do not repeat the same call."
            )
            _append_guard_tool_results(messages, tool_calls, correction)
            yield {
                "type": "model_redirected",
                "reason": "repeated_tool_call",
                "message": correction,
                "streak": identical_tool_streak,
            }
            continue

        for call in tool_calls:
            name = call.get("function", {}).get("name", "unknown")
            yield {"type": "tool_call", "name": name, "input": _tool_input(call)}

        tool_results = await executor.execute_all(tool_calls, context)
        for result in tool_results:
            yield {
                "type": "tool_result",
                "name": _tool_name_by_id(tool_calls, result.tool_use_id or ""),
                "result": result.content,
                "is_error": result.is_error,
            }
            messages.append(
                Message(
                    role="tool",
                    content=result.content,
                    tool_call_id=result.tool_use_id,
                )
            )

        if tool_results and all(result.is_error for result in tool_results):
            consecutive_tool_error_turns += 1
        else:
            consecutive_tool_error_turns = 0
        if error_limit and consecutive_tool_error_turns >= error_limit:
            termination_reason = "consecutive_tool_errors"
            termination_message = (
                f"Tool execution failed for {consecutive_tool_error_turns} consecutive turns."
            )
            yield _run_stopped_event(
                termination_reason,
                termination_message,
                turn,
                total_tokens,
            )
            break

    yield {
        "type": "done",
        "total_turns": turn,
        "total_tokens": total_tokens,
        "termination_reason": termination_reason,
        "termination_message": termination_message,
        "completed": termination_reason == "completed",
        "messages": messages,
    }


def _positive_int(value: Any, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _optional_positive_int(value: Any) -> int:
    try:
        parsed = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed > 0 else 0


def _optional_positive_float(value: Any) -> float:
    try:
        parsed = float(value or 0)
    except (TypeError, ValueError):
        return 0.0
    return parsed if parsed > 0 else 0.0


def _tool_batch_signature(calls: list[dict[str, Any]]) -> str:
    payload = [
        [str(call.get("function", {}).get("name") or "unknown"), _tool_input(call)]
        for call in calls
    ]
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _reached_budgets(
    *,
    turn: int,
    turn_limit: int,
    total_tokens: int,
    token_budget: int,
) -> list[str]:
    reasons: list[str] = []
    if turn >= turn_limit:
        reasons.append("max_turns")
    if token_budget and total_tokens >= token_budget:
        reasons.append("token_budget")
    return reasons


def _budget_extension_request(
    reasons: list[str],
    *,
    turn: int,
    turn_limit: int,
    total_tokens: int,
    token_budget: int,
    extension_turns: int,
    extension_tokens: int,
) -> dict[str, Any]:
    return {
        "reason": "+".join(reasons),
        "turn": turn,
        "turn_limit": turn_limit,
        "total_tokens": total_tokens,
        "token_budget": token_budget,
        "suggested_additional_turns": extension_turns if "max_turns" in reasons else 0,
        "suggested_additional_tokens": extension_tokens if "token_budget" in reasons else 0,
        "context_preserved": True,
    }


async def _resolve_budget_extension(callback, request: dict[str, Any]) -> dict[str, int] | None:
    if callback is None:
        return None
    decision = callback(request)
    if isawaitable(decision):
        decision = await decision
    if isinstance(decision, str):
        approved = decision.strip().lower() in {"y", "yes", "continue", "approve"}
        decision = {"continue": approved}
    elif isinstance(decision, bool):
        decision = {"continue": decision}
    if not isinstance(decision, dict) or not decision.get("continue"):
        return None
    return {
        "additional_turns": _optional_positive_int(
            decision.get("additional_turns") or request["suggested_additional_turns"]
        ),
        "additional_tokens": _optional_positive_int(
            decision.get("additional_tokens") or request["suggested_additional_tokens"]
        ),
    }


async def _run_stop_hook(
    callback,
    *,
    llm_client: LlmClient,
    original_request: str,
    proposed_answer: str,
    messages: list[Message],
) -> StopHookResult:
    if callback is None:
        return await verify_answer(
            llm_client=llm_client,
            original_request=original_request,
            proposed_answer=proposed_answer,
            messages=messages,
        )
    request = {
        "original_request": original_request,
        "proposed_answer": proposed_answer,
        "messages": messages,
    }
    result = callback(request)
    if isawaitable(result):
        result = await result
    if isinstance(result, StopHookResult):
        return result
    if isinstance(result, bool):
        return StopHookResult(approved=result, feedback="")
    if isinstance(result, dict):
        return StopHookResult(
            approved=bool(result.get("approved")),
            feedback=str(result.get("feedback") or ""),
        )
    return StopHookResult(
        approved=False,
        feedback="Custom Stop Hook returned an invalid verdict; verify the result explicitly.",
    )


def _append_guard_tool_results(
    messages: list[Message],
    calls: list[dict[str, Any]],
    reason: str,
) -> None:
    for call in calls:
        messages.append(
            Message(
                role="tool",
                content=f"SmartCLI safety guard: {reason}",
                tool_call_id=str(call.get("id") or ""),
            )
        )


def _run_stopped_event(
    reason: str,
    message: str,
    turn: int,
    total_tokens: int,
) -> dict[str, Any]:
    return {
        "type": "run_stopped",
        "reason": reason,
        "message": message,
        "turn": turn,
        "total_tokens": total_tokens,
    }


def _merge_tool_delta(tool_states: dict[int, dict[str, Any]], delta: dict[str, Any]) -> None:
    index = int(delta.get("index") or 0)
    state = tool_states.setdefault(
        index,
        {
            "id": delta.get("id") or f"tool_{index}",
            "type": "function",
            "function": {"name": "", "arguments": ""},
        },
    )
    if delta.get("id"):
        state["id"] = delta["id"]
    function = delta.get("function") or {}
    if function.get("name"):
        state["function"]["name"] = function["name"]
    if function.get("arguments"):
        state["function"]["arguments"] += function["arguments"]


def _finalize_tool_calls(tool_states: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    calls = []
    for index in sorted(tool_states):
        state = tool_states[index]
        if state["function"]["name"]:
            calls.append(state)
    return calls


def _tool_input(call: dict[str, Any]) -> dict[str, Any]:
    raw = call.get("function", {}).get("arguments") or "{}"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _tool_name_by_id(calls: list[dict[str, Any]], tool_call_id: str) -> str:
    for call in calls:
        if call.get("id") == tool_call_id:
            return str(call.get("function", {}).get("name") or "unknown")
    return "unknown"


def _prepend_skill_context(user_message: str, skill_context_buffer) -> str:
    if not skill_context_buffer or skill_context_buffer.is_empty():
        return user_message
    drained = skill_context_buffer.drain()
    if not drained:
        return user_message
    return f"{drained}\n\n---\nUser request:\n{user_message}"
