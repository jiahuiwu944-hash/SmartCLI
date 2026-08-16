from __future__ import annotations

import json
import re
import shlex
from contextlib import suppress
from dataclasses import dataclass, field
from inspect import isawaitable
from pathlib import Path
from typing import Any, Literal

from paicli.policy import AuditLog
from paicli.tools.base import Tool, ToolContext, ToolResult

HookBehavior = Literal["continue", "deny", "skip"]


@dataclass(slots=True)
class ToolHookContext:
    tool_call_id: str
    tool_name: str
    tool: Tool
    input_data: dict[str, Any]
    runtime: ToolContext
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PreToolDecision:
    behavior: HookBehavior = "continue"
    updated_input: dict[str, Any] | None = None
    message: str = ""


class ToolLifecycleHook:
    priority = 100

    async def before_tool(self, context: ToolHookContext) -> PreToolDecision | None:
        return None

    async def after_tool(
        self,
        context: ToolHookContext,
        result: ToolResult,
    ) -> ToolResult | None:
        return None

    async def on_tool_error(
        self,
        context: ToolHookContext,
        error: Exception,
    ) -> ToolResult | None:
        return None


class ToolHookManager:
    def __init__(self, hooks: list[ToolLifecycleHook] | None = None):
        self._hooks: list[ToolLifecycleHook] = []
        for hook in hooks or []:
            self.register(hook)

    def register(self, hook: ToolLifecycleHook) -> None:
        self._hooks.append(hook)
        self._hooks.sort(key=lambda item: item.priority)

    async def run_before(self, context: ToolHookContext) -> PreToolDecision:
        for hook in self._hooks:
            decision = hook.before_tool(context)
            if isawaitable(decision):
                decision = await decision
            if decision is None:
                continue
            if decision.updated_input is not None:
                if "approval_decision" in context.metadata:
                    raise RuntimeError("a pre-tool hook cannot modify input after approval")
                context.input_data = decision.updated_input
            if decision.behavior != "continue":
                return decision
        return PreToolDecision()

    async def run_after(
        self,
        context: ToolHookContext,
        result: ToolResult,
    ) -> ToolResult:
        current = result
        for hook in self._hooks:
            updated = hook.after_tool(context, current)
            if isawaitable(updated):
                updated = await updated
            if updated is not None:
                current = updated
        return current

    async def run_error(
        self,
        context: ToolHookContext,
        error: Exception,
    ) -> ToolResult | None:
        handled: ToolResult | None = None
        for hook in self._hooks:
            result = hook.on_tool_error(context, error)
            if isawaitable(result):
                result = await result
            if handled is None and result is not None:
                handled = result
        return handled


class ExplorationGuardHook(ToolLifecycleHook):
    """Nudge long-running diagnostic loops back toward an implementation."""

    priority = 50

    async def before_tool(self, context: ToolHookContext) -> PreToolDecision | None:
        limit = max(0, int(context.runtime.config.agent.exploration_tool_call_limit or 0))
        if not limit:
            return None
        cost = _exploration_cost(context)
        if not cost:
            return None
        runtime = context.runtime
        if runtime.exploration_next_warning <= 0:
            runtime.exploration_next_warning = limit
        runtime.exploration_call_count += cost
        if runtime.exploration_call_count < runtime.exploration_next_warning:
            return None
        context.metadata["exploration_warning"] = (
            "SmartCLI exploration guard: diagnostic activity has reached "
            f"{runtime.exploration_call_count} weighted calls. Summarize the evidence already "
            "collected and converge: use search_code/document_symbols for targeted navigation, "
            "read_file only for the required range, modify source directly with write_file, then "
            "run diagnose_file or the relevant tests. Do not create another probe unless the "
            "existing evidence cannot answer a specific question."
        )
        runtime.exploration_next_warning += limit
        return None

    async def after_tool(
        self,
        context: ToolHookContext,
        result: ToolResult,
    ) -> ToolResult | None:
        warning = str(context.metadata.get("exploration_warning") or "")
        if warning:
            result.content = f"{result.content}\n\n[{warning}]"
        return result


class ManagedScratchHook(ToolLifecycleHook):
    """Keep model-created probe files isolated so the runtime can clean them up."""

    priority = 75

    async def before_tool(self, context: ToolHookContext) -> PreToolDecision | None:
        if context.tool_name != "write_file":
            return None
        raw_path = str(context.input_data.get("path") or "")
        if not _looks_like_scratch_path(raw_path):
            return None
        if not _is_managed_scratch_path(raw_path, context.runtime.cwd):
            suggested = f".paicli/tmp/{Path(raw_path).name or 'probe.py'}"
            return PreToolDecision(
                behavior="deny",
                message=(
                    "Temporary probe files must be created under .paicli/tmp so SmartCLI can "
                    f"remove them when the run ends. Retry write_file with path {suggested!r}."
                ),
            )
        content = str(context.input_data.get("content") or "")
        if _content_mutates_protected_source(content):
            return PreToolDecision(
                behavior="deny",
                message=(
                    "A temporary script may not patch workspace source files. Read the target "
                    "source and apply the change directly with write_file and expected_version."
                ),
            )
        return None

    async def after_tool(
        self,
        context: ToolHookContext,
        result: ToolResult,
    ) -> ToolResult | None:
        if (
            context.tool_name == "write_file"
            and not result.is_error
            and _is_managed_scratch_path(
                str(context.input_data.get("path") or ""), context.runtime.cwd
            )
        ):
            path = _resolve_workspace_path(
                str(context.input_data["path"]), context.runtime.cwd
            )
            context.runtime.scratch_files.add(path)
        return None


class ShellSourceWriteGuardHook(ToolLifecycleHook):
    """Prevent shell commands from bypassing versioned, atomic source writes."""

    priority = 100

    async def before_tool(self, context: ToolHookContext) -> PreToolDecision | None:
        if context.tool_name not in {"bash", "execute_command"}:
            return None
        if not context.runtime.config.tools.shell_source_write_guard:
            return None
        command = str(context.input_data.get("command") or "")
        reason = _shell_source_mutation_reason(command, context.runtime.cwd)
        if not reason:
            return None
        return PreToolDecision(
            behavior="deny",
            message=(
                f"Shell source-write guard blocked this command ({reason}). bash is for running "
                "tests, diagnostics, and read-only inspection; modify workspace source through "
                "read_file followed by write_file with expected_version."
            ),
        )


class ApprovalHook(ToolLifecycleHook):
    priority = 1_000

    async def before_tool(self, context: ToolHookContext) -> PreToolDecision | None:
        mode = context.runtime.config.policy.hitl_mode
        if mode == "never" or (mode == "auto" and not context.tool.requires_approval):
            context.metadata["approver"] = "none"
            context.metadata["approval_decision"] = "approve"
            return None

        callback = context.runtime.approval_callback
        if callback is None:
            decision = "deny"
        else:
            decision = callback(
                {
                    "tool_name": context.tool.name,
                    "input": context.input_data,
                    "danger_level": context.tool.danger_level,
                    "description": context.tool.description,
                }
            )
            if isawaitable(decision):
                decision = await decision
            if decision not in {"approve", "deny", "skip"}:
                decision = "deny"

        context.metadata["approver"] = "hitl"
        context.metadata["approval_decision"] = decision
        if decision in {"deny", "skip"}:
            action = "denied" if decision == "deny" else "skipped"
            return PreToolDecision(
                behavior=decision,
                message=f'Tool "{context.tool.name}" was {action} by approval policy.',
            )
        return None


class AuditHook(ToolLifecycleHook):
    priority = 2_000

    async def after_tool(
        self,
        context: ToolHookContext,
        result: ToolResult,
    ) -> ToolResult | None:
        decision = str(context.metadata.get("approval_decision") or "approve")
        should_record = decision in {"deny", "skip"} or (
            not context.tool.is_read_only and context.runtime.config.features.audit_log
        )
        if should_record:
            warning = _result_warning(result)
            outcome = (
                decision
                if decision in {"deny", "skip"}
                else ("error" if result.is_error else ("warning" if warning else "allow"))
            )
            self._record(
                context,
                outcome,
                details={"warning": warning} if warning else None,
            )
        return None

    async def on_tool_error(
        self,
        context: ToolHookContext,
        error: Exception,  # noqa: ARG002
    ) -> ToolResult | None:
        if not context.tool.is_read_only and context.runtime.config.features.audit_log:
            self._record(context, "error")
        return None

    @staticmethod
    def _record(
        context: ToolHookContext,
        outcome: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        with suppress(Exception):
            AuditLog(context.runtime.config.policy.audit_log_path).record(
                tool_name=context.tool.name,
                input_data=context.input_data,
                outcome=outcome,
                approver=str(context.metadata.get("approver") or "none"),
                cwd=context.runtime.cwd,
                details=details,
            )


class CodeIndexRefreshHook(ToolLifecycleHook):
    """Keep the code navigation index aligned with successful workspace writes."""

    priority = 2_500

    async def after_tool(
        self,
        context: ToolHookContext,
        result: ToolResult,
    ) -> ToolResult | None:
        navigator = context.runtime.code_navigator
        if navigator is None or result.is_error:
            return None
        with suppress(Exception):
            if context.tool_name == "write_file" and context.input_data.get("path"):
                navigator.refresh_file(str(context.input_data["path"]))
            elif context.tool_name in {"bash", "execute_command"}:
                navigator.update()
        return None


class ErrorResultHook(ToolLifecycleHook):
    priority = 3_000

    async def on_tool_error(
        self,
        context: ToolHookContext,
        error: Exception,
    ) -> ToolResult | None:
        return ToolResult(
            tool_use_id=context.tool_call_id,
            content=f'Tool "{context.tool_name}" execution error: {error}',
            is_error=True,
        )


def default_tool_hooks() -> ToolHookManager:
    return ToolHookManager(
        [
            ExplorationGuardHook(),
            ManagedScratchHook(),
            ShellSourceWriteGuardHook(),
            ApprovalHook(),
            AuditHook(),
            CodeIndexRefreshHook(),
            ErrorResultHook(),
        ]
    )


def cleanup_managed_scratch(context: ToolContext) -> list[str]:
    """Remove only scratch files created and tracked during the current run."""

    removed: list[str] = []
    scratch_root = (Path(context.cwd).resolve() / ".paicli" / "tmp").resolve()
    for path in list(context.scratch_files):
        with suppress(ValueError):
            path.resolve().relative_to(scratch_root)
            with suppress(OSError):
                path.unlink(missing_ok=True)
                removed.append(str(path.relative_to(Path(context.cwd).resolve())))
        context.scratch_files.discard(path)
    with suppress(OSError):
        scratch_root.rmdir()
    return removed


def _result_warning(result: ToolResult) -> str:
    try:
        payload = json.loads(result.content)
    except (TypeError, json.JSONDecodeError):
        return ""
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("warning") or "")


def _exploration_cost(context: ToolHookContext) -> int:
    if context.tool_name in {"bash", "execute_command"}:
        return 1
    if context.tool_name == "write_file" and _looks_like_scratch_path(
        str(context.input_data.get("path") or "")
    ):
        return 1
    if context.tool_name != "read_file":
        return 0
    signature = json.dumps(
        {
            "path": str(context.input_data.get("path") or ""),
            "offset": int(context.input_data.get("offset") or 0),
            "limit": int(context.input_data.get("limit") or 0),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    if signature in context.runtime.explored_read_ranges:
        return 1
    context.runtime.explored_read_ranges.add(signature)
    return 0


def _looks_like_scratch_path(value: str) -> bool:
    name = Path(value.replace("\\", "/")).name.lower()
    return (
        name.startswith(("tmp_", "temp_", "probe_"))
        or "_probe." in name
        or name.endswith((".tmp", ".temp"))
    )


def _is_managed_scratch_path(value: str, cwd: str) -> bool:
    path = _resolve_workspace_path(value, cwd)
    root = Path(cwd).resolve() / ".paicli" / "tmp"
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _resolve_workspace_path(value: str, cwd: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path(cwd).resolve() / path).resolve()


def _shell_source_mutation_reason(command: str, cwd: str) -> str:
    lowered = command.lower()
    if re.search(r"\bgit\s+(?:apply|checkout|restore|reset|clean)\b", lowered):
        return "git workspace mutation"
    if re.search(r"(?:^|[;&|]\s*)patch(?:\.exe)?\s", lowered):
        return "patch command"
    if _content_mutates_protected_source(command):
        return "direct shell or inline-script source modification"

    for script_name in _directly_executed_scripts(command):
        script = _resolve_workspace_path(script_name, cwd)
        try:
            script.relative_to(Path(cwd).resolve())
        except ValueError:
            continue
        if not script.is_file():
            continue
        with suppress(OSError, UnicodeDecodeError):
            if _content_mutates_protected_source(script.read_text(encoding="utf-8")):
                return f"source-mutating script {script.name}"
    return ""


def _directly_executed_scripts(command: str) -> list[str]:
    """Return script entrypoints, excluding data/target paths passed to tools.

    In ``python -m pytest tests/test_demo.py``, the test file is an argument to
    pytest rather than a Python entrypoint and must not be inspected as a
    source-mutating script.
    """

    scripts: list[str] = []
    segments = re.split(r"&&|\|\||(?<!\|)\|(?!\|)|[;\r\n]+", command)
    for segment in segments:
        try:
            tokens = [_unquote_shell_token(token) for token in shlex.split(segment, posix=False)]
        except ValueError:
            continue
        tokens = [token for token in tokens if token]
        if tokens and tokens[0] in {"&", "call"}:
            tokens = tokens[1:]
        if not tokens:
            continue

        executable = Path(tokens[0]).name.lower()
        if _is_python_executable(executable):
            script = _python_entrypoint(tokens[1:])
        elif executable in {"powershell", "powershell.exe", "pwsh", "pwsh.exe"}:
            script = _powershell_entrypoint(tokens[1:])
        elif executable in {"bash", "bash.exe", "sh", "sh.exe"}:
            script = _first_script_argument(tokens[1:], {".sh"})
        elif executable in {"cmd", "cmd.exe"}:
            script = _cmd_entrypoint(tokens[1:])
        elif Path(tokens[0]).suffix.lower() in {".py", ".ps1", ".bat", ".cmd", ".sh"}:
            script = tokens[0]
        else:
            script = None
        if script:
            scripts.append(script)
    return scripts


def _unquote_shell_token(token: str) -> str:
    if len(token) >= 2 and token[0] == token[-1] and token[0] in {'"', "'"}:
        return token[1:-1]
    return token


def _is_python_executable(name: str) -> bool:
    return bool(re.fullmatch(r"(?:python(?:\d+(?:\.\d+)*)?|py)(?:\.exe)?", name))


def _python_entrypoint(arguments: list[str]) -> str | None:
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument in {"-m", "-c"} or argument.startswith(("-m=", "-c=")):
            return None
        if argument in {"-W", "-X"}:
            index += 2
            continue
        if argument.startswith("-"):
            index += 1
            continue
        return argument if Path(argument).suffix.lower() == ".py" else None
    return None


def _powershell_entrypoint(arguments: list[str]) -> str | None:
    for index, argument in enumerate(arguments[:-1]):
        if argument.lower() in {"-file", "/file"}:
            candidate = arguments[index + 1]
            return candidate if Path(candidate).suffix.lower() == ".ps1" else None
    return _first_script_argument(arguments, {".ps1"})


def _cmd_entrypoint(arguments: list[str]) -> str | None:
    for index, argument in enumerate(arguments[:-1]):
        if argument.lower() in {"/c", "/k"}:
            candidate = arguments[index + 1]
            return candidate if Path(candidate).suffix.lower() in {".bat", ".cmd"} else None
    return None


def _first_script_argument(arguments: list[str], suffixes: set[str]) -> str | None:
    for argument in arguments:
        if argument.startswith("-"):
            continue
        return argument if Path(argument).suffix.lower() in suffixes else None
    return None


def _content_mutates_protected_source(content: str) -> bool:
    lowered = content.lower()
    mutation_patterns = (
        r"\b(?:write_text|write_bytes|unlink|rename|replace)\s*\(",
        r"\bopen\s*\([^\n]{0,300},\s*[\"'][wax+]",
        r"\b(?:set-content|add-content|out-file|remove-item|move-item|copy-item)\b",
        r"\b(?:sed\s+-i|perl\s+-pi)\b",
        r"(?:^|[;&|]\s*)(?:del|erase|rm|mv|move|cp|copy)\s+",
        r">{1,2}\s*[^&|\r\n]+",
    )
    if not any(re.search(pattern, lowered, re.DOTALL) for pattern in mutation_patterns):
        return False
    protected_path_patterns = (
        r"(?:^|[\s\"'])(?:src|tests|app|lib)[\\/]",
        r"[\"'][^\"']*\.(?:py|pyi|js|jsx|ts|tsx|java|go|rs|c|cc|cpp|h|hpp|cs|rb|php|kt|kts)[\"']",
        r"\b(?:pyproject\.toml|package\.json|pom\.xml|build\.gradle)\b",
    )
    if not any(re.search(pattern, lowered) for pattern in protected_path_patterns):
        return False
    return not _only_managed_or_named_scratch_paths(lowered)


def _only_managed_or_named_scratch_paths(content: str) -> bool:
    code_paths = re.findall(
        r"(?i)([.\w/\\-]+\.(?:py|pyi|js|jsx|ts|tsx|java|go|rs|c|cc|cpp|h|hpp|cs|rb|php|kt|kts))",
        content,
    )
    return bool(code_paths) and all(
        ".paicli/tmp/" in path.replace("\\", "/").lower()
        or _looks_like_scratch_path(path)
        for path in code_paths
    )
