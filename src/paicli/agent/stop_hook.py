from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
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
5. A successful read_file evidence row includes the exact returned line range. Treat that
   range as proof that the agent inspected it; the full source body is intentionally omitted
   from the reviewer payload and must not be demanded again.

Give concrete feedback that tells the agent what to inspect, change, clarify, or test next.
Do not solve the task yourself and do not approve merely because the answer sounds plausible.

When and only when the task is approved, optionally extract at most 3 stable facts that would
be useful in a future session. Good memories include explicit user preferences, non-obvious
project constraints, verified decisions, and reusable debugging solutions. Do not remember
secrets, guesses, raw tool output, temporary progress, or facts easily read from source code.
Every fact or solution must cite one or more successful tool call IDs. Use a stable dotted key
so a future correction can replace the old value.

Return JSON only:
{
  "approved": true,
  "feedback": "",
  "memories": [
    {
      "key": "project.test_framework",
      "category": "decision",
      "content": "The project uses pytest for tests.",
      "importance": 0.8,
      "confidence": 0.95,
      "evidence": "The verified test command completed successfully.",
      "evidence_ids": ["tool_call_id"]
    }
  ]
}
For rejection return:
{"approved": false, "feedback": "specific next action", "memories": []}
"""


@dataclass(slots=True)
class StopHookResult:
    approved: bool
    feedback: str
    input_tokens: int = 0
    output_tokens: int = 0
    raw: str = ""
    memory_candidates: list[dict[str, Any]] = field(default_factory=list)


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
        memory_candidates=_parse_memory_candidates(text) if verdict[0] else [],
    )


def _parse_verdict(text: str) -> tuple[bool, str]:
    candidate = text.strip()
    verdicts = _decoded_verdicts(candidate)
    if verdicts:
        data = verdicts[-1]
        feedback = str(data.get("feedback") or "").strip()
        if not data["approved"] and not feedback:
            feedback = "The result was not verified. Re-check the task and provide evidence."
        return data["approved"], feedback

    # Some compatible models occasionally emit a JSON-like object with an
    # unescaped newline in feedback. Preserve the boolean verdict rather than
    # exhausting correction attempts solely because that string is malformed.
    approval_matches = list(
        re.finditer(r'["\']?approved["\']?\s*:\s*(true|false)\b', candidate, re.IGNORECASE)
    )
    if approval_matches:
        match = approval_matches[-1]
        approved = match.group(1).lower() == "true"
        feedback = _fallback_feedback(candidate[match.end() :])
        if not approved and not feedback:
            feedback = "The result was not verified. Re-check the task and provide evidence."
        return approved, feedback
    return False, "Stop Hook could not parse its verdict; verify the result explicitly."


def _decoded_verdicts(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    verdicts: list[dict[str, Any]] = []
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("approved"), bool):
            verdicts.append(value)
    return verdicts


def _parse_memory_candidates(text: str) -> list[dict[str, Any]]:
    verdicts = _decoded_verdicts(text.strip())
    if not verdicts or not verdicts[-1].get("approved"):
        return []
    memories = verdicts[-1].get("memories") or []
    if not isinstance(memories, list):
        return []
    return [item for item in memories[:3] if isinstance(item, dict)]


def _fallback_feedback(text: str) -> str:
    match = re.search(
        r'["\']?feedback["\']?\s*:\s*["\'](.*?)["\']\s*[,}]',
        text,
        re.IGNORECASE | re.DOTALL,
    )
    return match.group(1).replace("\\n", "\n").strip() if match else ""


def _recent_tool_evidence(messages: list[Message], *, max_chars: int = 12_000) -> str:
    entries: list[dict[str, str]] = []
    entries_by_id: dict[str, dict[str, str]] = {}
    for message in messages:
        if message.role == "assistant" and message.tool_calls:
            for call in message.tool_calls:
                function = call.get("function") or {}
                call_id = str(call.get("id") or "?")
                entry = {
                    "id": call_id,
                    "name": str(function.get("name") or "unknown"),
                    "arguments": str(function.get("arguments") or "{}"),
                    "status": "missing_result",
                    "result": "",
                }
                entries.append(entry)
                entries_by_id[call_id] = entry
        elif message.role == "tool":
            content = (
                message.content
                if isinstance(message.content, str)
                else json.dumps(message.content, ensure_ascii=False)
            )
            call_id = str(message.tool_call_id or "?")
            entry = entries_by_id.get(call_id)
            if entry is None:
                entry = {
                    "id": call_id,
                    "name": "unknown",
                    "arguments": "{}",
                    "status": "missing_result",
                    "result": "",
                }
                entries.append(entry)
                entries_by_id[call_id] = entry
            entry["status"] = "blocked" if _is_guard_result(content) else "returned"
            entry["result"] = content

    rows = [_format_evidence_entry(entry) for entry in entries]
    if not rows:
        return ""

    # Keep initial intent/evidence (for example load_skill and the first attempted
    # strategy) as well as the most recent verification. Tail-only truncation made
    # long runs falsely appear to have never loaded a skill or tried the requested tool.
    selected_indices: set[int] = set()
    used_chars = 0
    head_budget = max(1, max_chars // 3)
    for index, row in enumerate(rows[:4]):
        if selected_indices and used_chars + len(row) > head_budget:
            break
        selected_indices.add(index)
        used_chars += len(row)

    for index in range(len(rows) - 1, -1, -1):
        if index in selected_indices:
            continue
        row = rows[index]
        if selected_indices and used_chars + len(row) > max_chars:
            continue
        selected_indices.add(index)
        used_chars += len(row)
        if used_chars >= max_chars:
            break

    ordered = sorted(selected_indices)
    selected = [rows[index] for index in ordered]
    omitted = len(rows) - len(selected)
    if omitted:
        split = sum(1 for index in ordered if index < 4)
        selected.insert(split, f"[{omitted} middle tool evidence entries omitted]")
    return "\n\n---\n\n".join(selected)


def _format_evidence_entry(entry: dict[str, str]) -> str:
    result = _summarize_tool_result(
        entry["name"],
        entry["arguments"],
        entry["result"],
    )
    return (
        f"TOOL EVIDENCE id={entry['id']} name={entry['name']} "
        f"status={entry['status']} arguments={entry['arguments']}\n{result}"
    )


def _summarize_tool_result(name: str, arguments: str, content: str) -> str:
    if name == "read_file" and content:
        path_match = re.search(r"(?m)^path:\s*(.+?)\s*$", content)
        version_match = re.search(r"(?m)^version:\s*(.+?)\s*$", content)
        size_match = re.search(r"(?m)^size:\s*(\d+)\s*$", content)
        line_numbers = [
            int(value) for value in re.findall(r"(?m)^\s*(\d+)\s*:\s", content)
        ]
        try:
            parsed_arguments = json.loads(arguments)
        except json.JSONDecodeError:
            parsed_arguments = {}
        path = (
            path_match.group(1)
            if path_match
            else str(parsed_arguments.get("path") or "unknown")
        )
        fields = [f"read_succeeded path={path}"]
        if line_numbers:
            fields.append(f"returned_lines={min(line_numbers)}-{max(line_numbers)}")
        if size_match:
            fields.append(f"file_size={size_match.group(1)}")
        if version_match:
            fields.append(f"version={version_match.group(1)}")
        return " ".join(fields)

    if len(content) <= 900:
        return content
    return content[:620] + "\n... [result summarized] ...\n" + content[-220:]


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
    # Runtime-generated guard results always use this explicit prefix.  Do not scan
    # arbitrary tool output for phrases such as "tool call was blocked": a user may
    # legitimately read source code, tests, or documentation containing those words.
    return content.strip().lower().startswith("smartcli safety guard:")
