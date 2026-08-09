from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from paicli.llm.base import LlmClient
from paicli.types import Message

STOP_HOOK_SYSTEM_PROMPT = """You are SmartCLI's Stop Hook reviewer.
Before the agent is allowed to stop, decide whether its proposed final answer actually
completes the user's request and is supported by the available tool evidence.

Apply these rules strictly:
1. A blocked, skipped, denied, timed-out, or failed tool call is NOT a successful action.
2. Quantities and completion claims in the answer must match successful tool results.
3. Reject internal contradictions such as saying "all steps completed" while also saying
   that one step was blocked or not executed.
4. Reject unsupported claims, ignored errors, incomplete requirements, and code-changing
   tasks without appropriate verification evidence.

Give concrete feedback that tells the agent what to inspect, change, clarify, or test next.
Do not solve the task yourself and do not approve merely because the answer sounds plausible.

Return JSON only:
{"approved": true, "feedback": ""}
or
{"approved": false, "feedback": "specific next action"}
"""


@dataclass(slots=True)
class StopHookResult:
    approved: bool
    feedback: str
    input_tokens: int = 0
    output_tokens: int = 0
    raw: str = ""


async def verify_answer(
    *,
    llm_client: LlmClient,
    original_request: str,
    proposed_answer: str,
    messages: list[Message],
) -> StopHookResult:
    deterministic_feedback = _deterministic_contradiction(proposed_answer, messages)
    if deterministic_feedback:
        return StopHookResult(approved=False, feedback=deterministic_feedback)

    evidence = _recent_tool_evidence(messages)
    payload = (
        f"Original request:\n{original_request}\n\n"
        f"Proposed final answer:\n{proposed_answer or '(empty)'}\n\n"
        f"Recent tool evidence:\n{evidence or '(none)'}"
    )
    text = ""
    input_tokens = 0
    output_tokens = 0
    async for event in llm_client.chat(
        [Message(role="user", content=payload)],
        [],
        system_prompt=STOP_HOOK_SYSTEM_PROMPT,
    ):
        event_type = event.get("type")
        if event_type == "text_delta":
            text += str(event.get("text") or "")
        elif event_type == "usage":
            usage = event.get("usage") or {}
            input_tokens += int(usage.get("input_tokens") or 0)
            output_tokens += int(usage.get("output_tokens") or 0)
        elif event_type == "error":
            return StopHookResult(
                approved=False,
                feedback=f"Stop Hook failed: {event.get('error')}",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                raw=text,
            )

    verdict = _parse_verdict(text)
    return StopHookResult(
        approved=verdict[0],
        feedback=verdict[1],
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        raw=text,
    )


def _parse_verdict(text: str) -> tuple[bool, str]:
    candidate = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    else:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            candidate = candidate[start : end + 1]
    try:
        data: Any = json.loads(candidate)
    except json.JSONDecodeError:
        return False, "Stop Hook could not parse its verdict; verify the result explicitly."
    if not isinstance(data, dict) or not isinstance(data.get("approved"), bool):
        return False, "Stop Hook returned an invalid verdict; verify the result explicitly."
    feedback = str(data.get("feedback") or "").strip()
    if not data["approved"] and not feedback:
        feedback = "The result was not verified. Re-check the task and provide evidence."
    return data["approved"], feedback


def _recent_tool_evidence(messages: list[Message], *, max_chars: int = 12_000) -> str:
    rows: list[str] = []
    for message in reversed(messages):
        if message.role == "assistant" and message.tool_calls:
            for call in reversed(message.tool_calls):
                function = call.get("function") or {}
                rows.append(
                    "TOOL CALL "
                    f"id={call.get('id') or '?'} name={function.get('name') or 'unknown'} "
                    f"arguments={function.get('arguments') or '{}'}"
                )
        elif message.role == "tool":
            content = (
                message.content
                if isinstance(message.content, str)
                else json.dumps(message.content, ensure_ascii=False)
            )
            status = "blocked" if _is_guard_result(content) else "returned"
            rows.append(f"TOOL RESULT id={message.tool_call_id or '?'} status={status}\n{content}")
        else:
            continue
        if sum(len(row) for row in rows) >= max_chars:
            break
    return "\n\n---\n\n".join(reversed(rows))[-max_chars:]


def _deterministic_contradiction(answer: str, messages: list[Message]) -> str:
    has_blocked_result = any(
        message.role == "tool"
        and isinstance(message.content, str)
        and _is_guard_result(message.content)
        for message in messages
    )
    if not has_blocked_result:
        return ""

    normalized = answer.lower()
    explicit_incomplete = re.search(
        r"(?:原始)?(?:任务|请求|操作|读取)?(?:仍然|仍|尚|并)?未(?:全部)?完成|"
        r"没有完成|未能完成|部分完成|"
        r"(?:task|request)\s+(?:is\s+)?(?:not\s+complete|incomplete)|partially\s+completed",
        normalized,
    )
    if explicit_incomplete:
        return ""
    blocked_language = re.search(
        r"拦截|阻止|跳过|未执行|没有执行|失败|blocked|skipped|not\s+executed|failed",
        normalized,
    )
    completion_claim = re.search(
        r"(?:全部|所有|均已|已经|成功).{0,8}完成|(?<!未)已完成|"
        r"(?:\d+|[一二三四五六七八九十]+)次.{0,12}(?:已)?完成|"
        r"all\s+.{0,24}completed|successfully\s+completed|task\s+(?:is\s+)?complete",
        normalized,
    )
    if blocked_language and completion_claim:
        return (
            "The proposed answer contains a completion claim but also states that an action "
            "was blocked, skipped, failed, or not executed. Revise the answer so unsuccessful "
            "actions are not counted as completed, and clearly state whether the original "
            "request remains incomplete."
        )
    return ""


def _is_guard_result(content: str) -> bool:
    normalized = content.strip().lower()
    return normalized.startswith("smartcli safety guard:") or any(
        marker in normalized
        for marker in (
            "pending tool calls were not executed",
            "calls were not executed again",
            "tool call was blocked",
        )
    )
