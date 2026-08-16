from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from paicli.types import Message

from .manager import MemoryManager, MemoryMutation

_EVIDENCE_REQUIRED = {"fact", "project", "decision", "solution"}
_GROUNDING_STOP_WORDS = {
    "a",
    "an",
    "and",
    "for",
    "is",
    "of",
    "please",
    "the",
    "this",
    "to",
    "user",
    "with",
}
_FAILED_MARKERS = (
    '"is_error": true',
    '"error_code"',
    "execution error:",
    "smartcli safety guard:",
    "was denied by approval policy",
    "was skipped by approval policy",
)


@dataclass(slots=True)
class MemoryCaptureReport:
    mutations: list[MemoryMutation] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)

    @property
    def stored_ids(self) -> list[int]:
        return [item.memory_id for item in self.mutations]


def capture_approved_memories(
    manager: MemoryManager,
    candidates: list[dict[str, Any]],
    *,
    original_request: str,
    messages: list[Message],
    min_confidence: float = 0.8,
    max_candidates: int = 3,
) -> MemoryCaptureReport:
    """Persist verified candidates after completion approval.

    The LLM proposes memories, but deterministic gates own the write decision.
    """

    report = MemoryCaptureReport()
    successful_ids = _successful_tool_call_ids(messages)
    request_terms = _terms(original_request)
    for raw in candidates[: max(0, int(max_candidates))]:
        if not isinstance(raw, dict):
            report.rejected.append("candidate is not an object")
            continue
        content = _clean(raw.get("content"))
        memory_key = _clean(raw.get("key"))
        category = _clean(raw.get("category") or "fact").lower()
        evidence = _clean(raw.get("evidence"))
        evidence_ids = {
            str(value)
            for value in raw.get("evidence_ids", [])
            if isinstance(value, (str, int)) and str(value)
        }
        confidence = _number(raw.get("confidence"), 0.0)
        importance = _number(raw.get("importance"), 0.5)
        if not content or not memory_key:
            report.rejected.append("candidate is missing key or content")
            continue
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{1,119}", memory_key):
            report.rejected.append(f"{memory_key}: invalid stable key")
            continue
        if confidence < float(min_confidence):
            report.rejected.append(f"{memory_key}: confidence below threshold")
            continue
        if category in _EVIDENCE_REQUIRED and not evidence_ids.intersection(successful_ids):
            report.rejected.append(f"{memory_key}: no successful tool evidence")
            continue
        if category in {"preference", "constraint"}:
            content_terms = _terms(content)
            if not content_terms.intersection(request_terms) and not evidence_ids.intersection(
                successful_ids
            ):
                report.rejected.append(f"{memory_key}: not grounded in user request")
                continue
        if evidence_ids:
            evidence = (
                f"{evidence} [tool_call_ids={','.join(sorted(evidence_ids))}]"
            ).strip()
        try:
            mutation = manager.upsert(
                content,
                category=category,
                importance=importance,
                source="stop_hook",
                memory_key=memory_key,
                confidence=confidence,
                evidence=evidence,
            )
        except ValueError as exc:
            report.rejected.append(f"{memory_key}: {exc}")
            continue
        report.mutations.append(mutation)
    return report


def _successful_tool_call_ids(messages: list[Message]) -> set[str]:
    successful: set[str] = set()
    for message in messages:
        if message.role != "tool" or not message.tool_call_id:
            continue
        content = (
            message.content
            if isinstance(message.content, str)
            else json.dumps(message.content, ensure_ascii=False, default=str)
        )
        normalized = content.lower()
        if any(marker in normalized for marker in _FAILED_MARKERS):
            continue
        successful.add(str(message.tool_call_id))
    return successful


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _number(value: Any, default: float) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return default


def _terms(value: str) -> set[str]:
    normalized = str(value).lower()
    words = set(re.findall(r"[a-z_][a-z0-9_.-]*|\d+", normalized))
    chinese = re.findall(r"[\u4e00-\u9fff]+", normalized)
    for token in chinese:
        words.update(token[index : index + 2] for index in range(len(token) - 1))
    return {
        word.strip(".-")
        for word in words
        if word.strip(".-") and word.strip(".-") not in _GROUNDING_STOP_WORDS
    }
