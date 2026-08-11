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
        candidates: list[RunState] = []
        for path in self.directory.glob(f"{mode}-*.json"):
            state = self.load(path)
            if state and state.status == "PAUSED":
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
    return message.strip().lower() in {"continue", "resume", "继续", "继续执行", "/resume"}
