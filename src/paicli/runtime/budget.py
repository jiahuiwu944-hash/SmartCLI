from __future__ import annotations

from inspect import isawaitable
from typing import Any


class BudgetManager:
    """Shared turn/token budget gate for orchestrated Agent modes."""

    def __init__(self, *, turn_limit: int, token_limit: int, callback=None):
        self.turn_limit = max(1, turn_limit)
        self.token_limit = max(0, token_limit)
        self.turns = 0
        self.tokens = 0
        self.callback = callback

    def consume(self, *, turns: int = 0, tokens: int = 0) -> None:
        self.turns += max(0, turns)
        self.tokens += max(0, tokens)

    def reached(self, *, minimum_turns: int = 0) -> list[str]:
        reasons: list[str] = []
        if self.turns >= self.turn_limit or (
            minimum_turns and self.turns + minimum_turns > self.turn_limit
        ):
            reasons.append("max_turns")
        if self.token_limit and self.tokens >= self.token_limit:
            reasons.append("token_budget")
        return reasons

    @property
    def remaining_turns(self) -> int:
        return max(0, self.turn_limit - self.turns)

    @property
    def remaining_tokens(self) -> int:
        if not self.token_limit:
            return 0
        return max(0, self.token_limit - self.tokens)

    def can_start(self, *, minimum_turns: int = 1) -> bool:
        return self.remaining_turns >= max(1, minimum_turns) and not (
            self.token_limit and self.remaining_tokens <= 0
        )

    async def request_extension(
        self,
        *,
        additional_turns: int,
        additional_tokens: int,
        mode: str,
        minimum_turns: int = 0,
    ) -> tuple[bool, dict[str, Any]]:
        reasons = self.reached(minimum_turns=minimum_turns)
        request = {
            "reason": "+".join(reasons),
            "mode": mode,
            "turn": self.turns,
            "turn_limit": self.turn_limit,
            "total_tokens": self.tokens,
            "token_budget": self.token_limit,
            "suggested_additional_turns": additional_turns if "max_turns" in reasons else 0,
            "suggested_additional_tokens": (additional_tokens if "token_budget" in reasons else 0),
            "context_preserved": True,
        }
        if not reasons:
            return True, request
        if self.callback is None:
            return False, request
        decision = self.callback(request)
        if isawaitable(decision):
            decision = await decision
        if isinstance(decision, str):
            decision = decision.strip().lower() in {"y", "yes", "continue", "approve"}
        if isinstance(decision, bool):
            decision = {"continue": decision}
        if not isinstance(decision, dict) or not decision.get("continue"):
            return False, request
        added_turns = _positive(
            decision.get("additional_turns") or request["suggested_additional_turns"]
        )
        added_tokens = _positive(
            decision.get("additional_tokens") or request["suggested_additional_tokens"]
        )
        self.turn_limit += added_turns
        if added_tokens:
            self.token_limit = (self.token_limit or self.tokens) + added_tokens
        request.update(
            {
                "additional_turns": added_turns,
                "additional_tokens": added_tokens,
                "turn_limit": self.turn_limit,
                "token_budget": self.token_limit,
            }
        )
        return True, request


def _positive(value: Any) -> int:
    try:
        parsed = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)
