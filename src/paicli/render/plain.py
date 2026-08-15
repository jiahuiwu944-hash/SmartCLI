from __future__ import annotations

import sys
from typing import Any


class PlainRenderer:
    def __init__(self, *, print_events: bool = True):
        self.print_events = print_events

    def handle(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        if event_type == "text_delta":
            sys.stdout.write(str(event.get("text") or ""))
            sys.stdout.flush()
        elif self.print_events and event_type == "tool_call":
            sys.stdout.write(f"\n[tool] {event.get('name')} {event.get('input')}\n")
            sys.stdout.flush()
        elif self.print_events and event_type == "tool_result":
            marker = "error" if event.get("is_error") else "result"
            sys.stdout.write(f"[tool:{marker}] {event.get('name')}: {event.get('result')}\n")
            sys.stdout.flush()
        elif event_type == "run_stopped":
            reason = str(event.get("reason") or "safety_limit")
            message = str(event.get("message") or "Agent run stopped")
            sys.stdout.write(f"\n[stopped:{reason}] {message}\n")
            sys.stdout.flush()
        elif event_type == "stop_hook_review":
            status = "approved" if event.get("approved") else "revision-required"
            sys.stdout.write(f"\n[stop-hook:{status}] {event.get('feedback') or ''}\n")
            sys.stdout.flush()
        elif event_type == "model_redirected":
            sys.stdout.write(f"\n[redirected:{event.get('reason')}] {event.get('message') or ''}\n")
            sys.stdout.flush()
        elif event_type == "budget_extended":
            sys.stdout.write(
                "\n[budget-extended] "
                f"+{event.get('additional_turns') or 0} turns, "
                f"+{event.get('additional_tokens') or 0} tokens; context preserved.\n"
            )
            sys.stdout.flush()
        elif event_type == "context_compressed":
            before = float(event.get("before_pressure") or 0.0)
            after = float(event.get("after_pressure") or 0.0)
            sys.stdout.write(f"\n[context-compressed] {before:.0%} -> {after:.0%}\n")
            sys.stdout.flush()
        elif event_type == "error":
            message = str(event.get("message") or event.get("error") or "Unknown error")
            sys.stdout.write(f"\n[error:context-preserved] {message}\n")
            sys.stdout.flush()

    def newline(self) -> None:
        sys.stdout.write("\n")
        sys.stdout.flush()
