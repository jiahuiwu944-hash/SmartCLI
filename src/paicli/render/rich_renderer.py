from __future__ import annotations

import json
import re
from typing import Any

from rich import box
from rich.align import Align
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class RichRenderer:
    def __init__(
        self,
        console: Console | None = None,
        *,
        live_markdown: bool = False,
        context_window: int | None = None,
        show_thinking: bool = False,
    ):
        self.console = console or Console()
        self._buffer: list[str] = []
        self._thinking_buffer: list[str] = []
        self._pending_answer = ""
        self._show_thinking = show_thinking
        self._live_markdown = live_markdown
        self._live: Live | None = None
        self._thinking_live: Live | None = None
        self._context_window = context_window or 0
        self._input_tokens = 0
        self._output_tokens = 0
        self._last_input_tokens = 0
        self._last_turns = 0
        self._last_total_tokens = 0
        self._last_context_ratio = 0.0
        self._last_has_usage = False
        self._has_context_estimate = False
        self._context_details: dict[str, Any] = {}

    def set_context_window(self, context_window: int | None) -> None:
        self._context_window = context_window or self._context_window

    def start_run(self) -> None:
        self._buffer.clear()
        self._thinking_buffer.clear()
        self._pending_answer = ""
        self._stop_live_markdown()
        self._stop_live_thinking()
        self._input_tokens = 0
        self._output_tokens = 0
        self._last_input_tokens = 0
        self._has_context_estimate = False

    def toolbar_status(self) -> dict[str, Any]:
        return {
            "turns": self._last_turns,
            "input_tokens": self._input_tokens,
            "output_tokens": self._output_tokens,
            "total_tokens": self._last_total_tokens,
            "context_ratio": self._last_context_ratio,
            "has_usage": self._last_has_usage,
            **self._context_details,
        }

    def banner(
        self,
        *,
        model: str,
        provider: str,
        cwd: str,
        tools: int,
        version: str = "0.1.0",
        api_key_configured: bool = False,
        mcp_servers: int = 0,
        skills: int = 0,
        agents_files: int = 0,
        hitl_mode: str = "auto",
    ) -> None:
        top = Table.grid(expand=True)
        top.add_column(ratio=1)
        top.add_column(ratio=2)
        top.add_row(
            self._identity_panel(version=version, api_key_configured=api_key_configured),
            self._release_panel(version=version),
        )

        _ = model, provider, cwd, tools, mcp_servers, skills, agents_files, hitl_mode

        self.console.print()
        self.console.print(top)
        self.console.print(Align.right(Text("? for shortcuts", style="dim")))
        self.console.rule(style="grey23")
        self.console.print()

    def handle(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        if event_type == "text_delta":
            self._flush_thinking()
            text = str(event.get("text") or "")
            self._buffer.append(text)
            self._update_live_markdown()
        elif event_type == "thinking_delta":
            if self._show_thinking:
                thinking = str(event.get("thinking") or "")
                self._thinking_buffer.append(thinking)
                self._update_live_thinking()
        elif event_type == "usage":
            self._record_usage(event.get("usage") or {})
        elif event_type == "context_usage":
            self._record_context_usage(event)
        elif event_type == "context_compressed":
            self._print_context_compressed(event)
        elif event_type == "turn_complete":
            stop_reason = str(event.get("stop_reason") or "end_turn")
            self._flush_thinking()
            if stop_reason == "tool_use":
                self._discard_markdown()
            elif event.get("verification_pending"):
                self._hold_answer_for_verification()
            else:
                self._flush_markdown(title="Final Output")
        elif event_type == "tool_call":
            self._flush_thinking()
            self._flush_markdown(title="Assistant Output")
            self._print_tool_call(event)
        elif event_type == "tool_result":
            self._flush_thinking()
            self._flush_markdown(title="Assistant Output")
            self._print_tool_result(event)
        elif event_type == "stop_hook_review":
            self._flush_thinking()
            self._discard_markdown()
            self._print_stop_hook_review(event)
            if event.get("approved"):
                self._flush_pending_answer()
            else:
                self._pending_answer = ""
        elif event_type == "model_redirected":
            self._flush_thinking()
            self._flush_markdown(title="Assistant Output")
            self._print_model_redirected(event)
        elif event_type == "budget_extension_requested":
            self._flush_thinking()
            self._flush_markdown(title="Assistant Output")
        elif event_type == "budget_extended":
            self._print_budget_extended(event)
        elif event_type == "run_stopped":
            self._flush_thinking()
            self._flush_markdown(title="Assistant Output")
            self._print_run_stopped(event)
        elif event_type == "answer_preserved":
            self._flush_thinking()
            self._flush_markdown(title="Assistant Output")
            self._pending_answer = ""
            self._print_preserved_answer(event)
        elif event_type == "llm_retry":
            self._flush_thinking()
            self._flush_markdown(title="Assistant Output")
            self._print_llm_retry(event)
        elif event_type == "error":
            self._flush_thinking()
            self._flush_markdown(title="Assistant Output")
            message = str(event.get("message") or event.get("error") or "Unknown error")
            self.console.print(
                _output_panel(
                    message,
                    title=Text("Connection Error · context preserved", style="bold #ff4d5a"),
                    border_style="#ff4d5a",
                )
            )
        elif event_type == "done":
            self._flush_thinking()
            self._flush_pending_answer()
            self._flush_markdown(title="Final Output")
            self._record_run_summary(event)

    def markdown(self, text: str) -> None:
        self.console.print(Markdown(text))

    def newline(self) -> None:
        self._flush_thinking()
        self._flush_markdown(title="Final Output")
        self.console.print()

    def _flush_markdown(self, *, title: str) -> None:
        if not self._buffer:
            return
        text = "".join(self._buffer)
        self._buffer.clear()
        self._stop_live_markdown()
        if text.strip():
            self.console.print(
                _output_panel(
                    Markdown(text),
                    title=Text(title, style="bold #a8ff60"),
                    border_style="#3f3f46",
                )
            )

    def _discard_markdown(self) -> None:
        """Hide transitory narration such as 'I will inspect the file' before tool use."""
        self._buffer.clear()
        self._stop_live_markdown()

    def _hold_answer_for_verification(self) -> None:
        self._pending_answer = "".join(self._buffer).strip()
        self._buffer.clear()
        self._stop_live_markdown()

    def _flush_pending_answer(self) -> None:
        text = self._pending_answer.strip()
        self._pending_answer = ""
        if text:
            self.console.print(
                _output_panel(
                    Markdown(text),
                    title=Text("Final Output", style="bold #a8ff60"),
                    border_style="#3f3f46",
                )
            )

    def _update_live_markdown(self) -> None:
        if not self._live_markdown or not self.console.is_terminal:
            return
        text = "".join(self._buffer)
        if not text.strip():
            return
        renderable = _output_panel(
            Markdown(text),
            title=Text("Assistant Output", style="bold #a8ff60"),
            border_style="#3f3f46",
        )
        if self._live is None:
            self._live = Live(
                renderable,
                console=self.console,
                refresh_per_second=12,
                transient=True,
                vertical_overflow="visible",
            )
            self._live.start(refresh=True)
            return
        self._live.update(renderable, refresh=True)

    def _stop_live_markdown(self) -> None:
        if self._live is None:
            return
        self._live.stop()
        self._live = None

    def _flush_thinking(self) -> None:
        if not self._thinking_buffer:
            return
        text = "".join(self._thinking_buffer)
        self._thinking_buffer.clear()
        self._stop_live_thinking()
        if text.strip():
            self.console.print(
                _output_panel(
                    Text(text, style="dim"),
                    title=Text("Thinking", style="bold #c084fc"),
                    border_style="#6d28d9",
                )
            )

    def _update_live_thinking(self) -> None:
        if not self._live_markdown or not self.console.is_terminal:
            return
        text = "".join(self._thinking_buffer)
        if not text.strip():
            return
        renderable = _output_panel(
            Text(text, style="dim"),
            title=Text("Thinking", style="bold #c084fc"),
            border_style="#6d28d9",
        )
        if self._thinking_live is None:
            self._thinking_live = Live(
                renderable,
                console=self.console,
                refresh_per_second=12,
                transient=True,
                vertical_overflow="visible",
            )
            self._thinking_live.start(refresh=True)
            return
        self._thinking_live.update(renderable, refresh=True)

    def _stop_live_thinking(self) -> None:
        if self._thinking_live is None:
            return
        self._thinking_live.stop()
        self._thinking_live = None

    def _record_usage(self, usage: dict[str, Any]) -> None:
        input_tokens = int(usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        self._input_tokens += input_tokens
        self._output_tokens += output_tokens
        if input_tokens:
            self._last_input_tokens = input_tokens

    def _record_context_usage(self, event: dict[str, Any]) -> None:
        self._last_context_ratio = float(event.get("pressure_ratio") or 0.0)
        self._last_has_usage = True
        self._has_context_estimate = True
        self._context_details = {
            "estimated_input_tokens": int(event.get("estimated_input_tokens") or 0),
            "context_window": int(event.get("context_window") or self._context_window),
            "output_reserve": int(event.get("output_reserve") or 0),
            "tool_reserve": int(event.get("tool_reserve") or 0),
            "safety_margin": int(event.get("safety_margin") or 0),
            "target_ratio": float(event.get("target_ratio") or 0.0),
            "pressure_level": str(event.get("pressure_level") or "normal"),
        }

    def _print_context_compressed(self, event: dict[str, Any]) -> None:
        before = float(event.get("before_pressure") or 0.0)
        after = float(event.get("after_pressure") or 0.0)
        tools = int(event.get("compacted_tool_results") or 0)
        messages = int(event.get("summarized_messages") or 0)
        mode = "emergency" if event.get("emergency") else "automatic"
        self.console.print(
            f"[yellow]Context compressed ({mode}):[/yellow] "
            f"{before:.0%} -> {after:.0%}; {tools} tool results slimmed, "
            f"{messages} messages summarized."
        )

    def _print_tool_call(self, event: dict[str, Any]) -> None:
        name = str(event.get("name") or "unknown")
        payload = event.get("input") or {}
        body = Table.grid(padding=(0, 1))
        body.add_column(style="dim", no_wrap=True)
        body.add_column()
        body.add_row("name", Text(name, style="bold #facc15"))
        body.add_row("input", Text(_format_payload(payload), style="#e5e7eb"))
        self.console.print(
            _output_panel(
                body,
                title=Text("Tool Use", style="bold #facc15"),
                border_style="#facc15",
            )
        )

    def _print_tool_result(self, event: dict[str, Any]) -> None:
        is_error = bool(event.get("is_error"))
        name = str(event.get("name") or "unknown")
        result = _tool_result_preview(
            name,
            str(event.get("result") or ""),
            is_error=is_error,
        )
        title_style = "bold #ff4d5a" if is_error else "bold #22c55e"
        border_style = "#ff4d5a" if is_error else "#22c55e"
        status = "error" if is_error else "ok"
        self.console.print(
            _output_panel(
                result or "(empty result)",
                title=Text(f"Tool Result · {name} · {status}", style=title_style),
                border_style=border_style,
            )
        )

    def _print_preserved_answer(self, event: dict[str, Any]) -> None:
        text = str(event.get("text") or "").strip()
        if not text:
            return
        self.console.print(
            _output_panel(
                Markdown(text),
                title=Text(
                    "Best Available Answer · verification incomplete",
                    style="bold #facc15",
                ),
                border_style="#facc15",
            )
        )

    def _print_run_stopped(self, event: dict[str, Any]) -> None:
        reason = str(event.get("reason") or "safety_limit")
        message = str(event.get("message") or "Agent run stopped")
        self.console.print(
            _output_panel(
                message,
                title=Text(f"Agent Stopped · {reason}", style="bold #fb923c"),
                border_style="#fb923c",
            )
        )

    def _print_llm_retry(self, event: dict[str, Any]) -> None:
        attempt = int(event.get("attempt") or 1)
        max_retries = int(event.get("max_retries") or 0)
        delay = float(event.get("delay_seconds") or 0.0)
        reason = str(event.get("reason") or "temporary API error")
        self.console.print(
            _output_panel(
                f"{reason}; retrying in {delay:g}s.",
                title=Text(
                    f"Model API Retry · {attempt}/{max_retries}",
                    style="bold #facc15",
                ),
                border_style="#facc15",
            )
        )

    def _print_stop_hook_review(self, event: dict[str, Any]) -> None:
        approved = bool(event.get("approved"))
        feedback = str(event.get("feedback") or "Answer verified.")
        if approved:
            self.console.print("[dim green]✓ Stop Hook verified[/dim green]")
            return
        status = "approved" if approved else "revision required"
        color = "#22c55e" if approved else "#fb923c"
        self.console.print(
            _output_panel(
                feedback,
                title=Text(f"Stop Hook · {status}", style=f"bold {color}"),
                border_style=color,
            )
        )

    def _print_model_redirected(self, event: dict[str, Any]) -> None:
        reason = str(event.get("reason") or "correction")
        message = str(event.get("message") or "The model was asked to revise its approach.")
        self.console.print(
            _output_panel(
                message,
                title=Text(f"Agent Correction · {reason}", style="bold #c084fc"),
                border_style="#6d28d9",
            )
        )

    def _print_budget_extended(self, event: dict[str, Any]) -> None:
        turns = int(event.get("additional_turns") or 0)
        tokens = int(event.get("additional_tokens") or 0)
        additions = []
        if turns:
            additions.append(f"+{turns} turns")
        if tokens:
            additions.append(f"+{tokens} tokens")
        self.console.print(
            f"[green]Budget extended:[/green] {', '.join(additions)}; context preserved."
        )

    def _record_run_summary(self, event: dict[str, Any]) -> None:
        total_tokens = int(event.get("total_tokens") or self._input_tokens + self._output_tokens)
        turns = int(event.get("total_turns") or 0)
        has_usage = total_tokens > 0 or self._input_tokens > 0 or self._output_tokens > 0
        context_ratio = self._last_context_ratio
        if not self._has_context_estimate:
            context_ratio = (
                self._last_input_tokens / self._context_window
                if self._context_window > 0
                else 0
            )
        self._last_turns = turns
        self._last_total_tokens = total_tokens
        self._last_context_ratio = context_ratio
        self._last_has_usage = has_usage

    def _identity_panel(self, *, version: str, api_key_configured: bool) -> Table:
        logo = Text("\n".join(_PI_LOGO), style="bold #a8ff60")
        identity = Text()
        identity.append("SmartCLI ", style="bold white")
        identity.append(f"v{version}", style="dim")
        identity.append("\n\n")
        if api_key_configured:
            identity.append("Signed in ", style="bold white")
            identity.append("API Key", style="dim")
        else:
            identity.append("Missing ", style="bold red")
            identity.append("API Key", style="dim")

        grid = Table.grid(padding=(0, 2))
        grid.add_column(no_wrap=True)
        grid.add_column()
        grid.add_row(logo, Align.center(identity, vertical="middle"))
        return grid

    def _release_panel(self, *, version: str) -> Panel:
        notes = Text()
        for line in [
            "π logo home layout for the interactive CLI",
            "MCP, skills, tools, and workspace status at a glance",
            "Use /help for commands and /config for runtime settings",
        ]:
            notes.append("- ", style="dim")
            notes.append(line, style="dim")
            notes.append("\n")
        notes.append("/help", style="purple")
        notes.append(" for more", style="dim")
        return Panel(
            notes,
            title=Text(f"What's new (v{version})", style="bold green"),
            border_style="grey37",
            box=box.ROUNDED,
            padding=(0, 2),
        )


_PI_LOGO = (
    "████████████",
    "  ██    ██  ",
    "  ██    ██  ",
    "  ██    ██  ",
    "  ██    ██  ",
    "  ██    ██  ",
)


def _format_payload(payload: Any) -> str:
    try:
        return json.dumps(payload, ensure_ascii=False, indent=2)
    except TypeError:
        return str(payload)


_QUIET_TOOL_RESULTS = {
    "read_file",
    "search_code",
    "repo_map",
    "document_symbols",
    "grep",
    "glob",
    "glob_files",
}


def _tool_result_preview(name: str, result: str, *, is_error: bool) -> str:
    if is_error or name not in _QUIET_TOOL_RESULTS:
        if len(result) > 1200:
            return result[:1200] + "\n... [truncated]"
        return result or "(empty result)"

    if name == "read_file":
        path_match = re.search(r"(?m)^path:\s*(.+?)\s*$", result)
        size_match = re.search(r"(?m)^size:\s*(\d+)\s*$", result)
        path = path_match.group(1) if path_match else "requested file"
        details = [f"Read {path}"]
        if size_match:
            details.append(f"{int(size_match.group(1)):,} bytes")
        details.append("latest source loaded into Agent context")
        return " · ".join(details) + "."

    line_count = len(result.splitlines())
    return (
        f"{line_count} lines / {len(result):,} characters returned to Agent context. "
        "Full result hidden from terminal."
    )


def _output_panel(renderable: Any, *, title: Text, border_style: str) -> Panel:
    return Panel(
        renderable,
        title=title,
        border_style=border_style,
        box=box.ROUNDED,
        expand=True,
    )
