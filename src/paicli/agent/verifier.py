from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from paicli.agent.stop_hook import verify_answer
from paicli.llm.base import LlmClient
from paicli.types import Message


@dataclass(slots=True)
class VerificationResult:
    approved: bool
    feedback: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    memory_candidates: list[dict[str, Any]] | None = None


class CompletionVerifier:
    """One completion contract with mode-specific verification depth."""

    def __init__(self, llm_client: LlmClient):
        self.llm_client = llm_client

    def verify_task(
        self,
        *,
        description: str,
        result: str,
        messages: list[Message],
    ) -> VerificationResult:
        if not result.strip():
            return VerificationResult(False, f'Task "{description}" returned no result.')
        tool_messages = [
            message
            for message in messages
            if message.role == "tool" and isinstance(message.content, str)
        ]
        failures = [
            message for message in tool_messages if _failed_tool_result(str(message.content))
        ]
        blocked = any("smartcli safety guard:" in str(item.content).lower() for item in failures)
        if blocked or (tool_messages and len(failures) == len(tool_messages)):
            return VerificationResult(
                False,
                "A tool failed or was blocked; the task cannot be marked COMPLETED until "
                "the failure is corrected or reported as an explicit blocker.",
            )
        return VerificationResult(True)

    async def verify_final(
        self,
        *,
        original_request: str,
        proposed_answer: str,
        messages: list[Message],
    ) -> VerificationResult:
        result = await verify_answer(
            llm_client=self.llm_client,
            original_request=original_request,
            proposed_answer=proposed_answer,
            messages=messages,
        )
        return VerificationResult(
            result.approved,
            result.feedback,
            result.input_tokens,
            result.output_tokens,
            result.memory_candidates,
        )


def _failed_tool_result(content: str) -> bool:
    normalized = content.lower()
    return any(
        marker in normalized
        for marker in (
            '"error_code"',
            '"is_error": true',
            "execution error:",
            "smartcli safety guard:",
            "was denied by approval policy",
            "was skipped by approval policy",
        )
    )
