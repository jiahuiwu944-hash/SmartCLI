from __future__ import annotations

import json
from contextlib import suppress
from dataclasses import dataclass, field
from inspect import isawaitable
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
    return ToolHookManager([ApprovalHook(), AuditHook(), CodeIndexRefreshHook(), ErrorResultHook()])


def _result_warning(result: ToolResult) -> str:
    try:
        payload = json.loads(result.content)
    except (TypeError, json.JSONDecodeError):
        return ""
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("warning") or "")
