from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class RunState:
    run_id: str
    mode: str
    goal: str
    status: str = "RUNNING"
    turns: int = 0
    tokens: int = 0
    payload: dict[str, Any] = field(default_factory=dict)
    updated_at: float = field(default_factory=time.time)
    schema_version: int = 2


class RunStateStore:
    """Atomic JSON checkpoints for resumable Plan and Team runs."""

    def __init__(self, root: str | Path):
        self.directory = Path(root).resolve() / ".paicli" / "runs"
        self.directory.mkdir(parents=True, exist_ok=True)

    def create(self, mode: str, goal: str) -> RunState:
        state = RunState(run_id=uuid.uuid4().hex[:12], mode=mode, goal=goal)
        self.save(state)
        return state

    def save(self, state: RunState) -> None:
        state.updated_at = time.time()
        target = self._path(state)
        temporary = target.with_suffix(f".{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(asdict(state), handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    def latest_paused(self, mode: str) -> RunState | None:
        return self.latest_resumable(mode, statuses={"PAUSED"})

    def latest_resumable(
        self,
        mode: str,
        *,
        run_id: str | None = None,
        statuses: set[str] | None = None,
    ) -> RunState | None:
        allowed = statuses or {"PAUSED", "RUNNING"}
        candidates: list[RunState] = []
        for path in self.directory.glob(f"{mode}-*.json"):
            state = self.load(path)
            if (
                state
                and state.status in allowed
                and (run_id is None or state.run_id == run_id)
            ):
                candidates.append(state)
        return max(candidates, key=lambda item: item.updated_at, default=None)

    def load(self, value: str | Path) -> RunState | None:
        path = Path(value)
        if not path.is_absolute():
            path = self.directory / path
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return RunState(**payload)
        except (OSError, json.JSONDecodeError, TypeError):
            return None

    def _path(self, state: RunState) -> Path:
        return self.directory / f"{state.mode}-{state.run_id}.json"


def is_resume_request(message: str) -> bool:
    return resume_target(message) is not False


def resume_target(message: str) -> str | None | bool:
    """Return an optional run id, or False when *message* is not a resume command."""
    parts = message.strip().split(maxsplit=1)
    if not parts or parts[0].lower() not in {
        "continue",
        "resume",
        "继续",
        "继续执行",
        "/resume",
    }:
        return False
    return parts[1].strip() if len(parts) == 2 and parts[1].strip() else None
