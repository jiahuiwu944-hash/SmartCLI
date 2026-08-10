from __future__ import annotations

import asyncio
from typing import Any

from paicli.tools.base import Tool, ToolContext, ToolResult
from paicli.tools.hooks import ToolHookContext, ToolHookManager, default_tool_hooks
from paicli.tools.registry import ToolRegistry


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        hook_manager: ToolHookManager | None = None,
    ):
        self.registry = registry
        self.hooks = hook_manager or default_tool_hooks()

    async def execute_all(
        self,
        calls: list[dict[str, Any]],
        context: ToolContext,
    ) -> list[ToolResult]:
        read_calls: list[tuple[dict[str, Any], Tool]] = []
        sequential_calls: list[tuple[dict[str, Any], Tool | None]] = []

        for call in calls:
            name = _tool_call_name(call)
            tool = self.registry.get(name)
            if tool and tool.is_read_only and tool.is_concurrency_safe:
                read_calls.append((call, tool))
            else:
                sequential_calls.append((call, tool))

        results: list[ToolResult] = []
        if read_calls:
            semaphore = asyncio.Semaphore(context.config.tools.max_concurrent_read)

            async def run_read(call: dict[str, Any], tool: Tool) -> ToolResult:
                async with semaphore:
                    return await self._execute_single(call, tool, context)

            results.extend(
                await asyncio.gather(*(run_read(call, tool) for call, tool in read_calls))
            )

        for call, tool in sequential_calls:
            results.append(await self._execute_single(call, tool, context))

        return results

    async def _execute_single(
        self,
        call: dict[str, Any],
        tool: Tool | None,
        context: ToolContext,
    ) -> ToolResult:
        tool_call_id = str(call.get("id") or "")
        name = _tool_call_name(call)
        payload = _tool_call_arguments(call)

        if not tool:
            return ToolResult(
                tool_use_id=tool_call_id,
                content=(
                    f'Tool "{name}" not found. Available tools: '
                    f"{', '.join(self.registry.list_names())}"
                ),
                is_error=True,
            )

        hook_context = ToolHookContext(
            tool_call_id=tool_call_id,
            tool_name=name,
            tool=tool,
            input_data=payload,
            runtime=context,
        )
        try:
            hook_context.input_data = tool.validate(payload)
            decision = await self.hooks.run_before(hook_context)
            if decision.behavior in {"deny", "skip"}:
                result = ToolResult(
                    tool_use_id=tool_call_id,
                    content=decision.message
                    or f'Tool "{tool.name}" was {decision.behavior}ed by a pre-tool hook.',
                    is_error=True,
                )
                return await self.hooks.run_after(hook_context, result)

            result = await tool.execute(hook_context.input_data, context)
            result.tool_use_id = tool_call_id
            result = await self.hooks.run_after(hook_context, result)
            result.tool_use_id = tool_call_id
            return result
        except Exception as exc:  # noqa: BLE001 - tool errors must flow back to the model
            result = await self.hooks.run_error(hook_context, exc)
            if result is None:
                result = ToolResult(
                    content=f'Tool "{name}" execution error: {exc}',
                    is_error=True,
                )
            result.tool_use_id = tool_call_id
            return result


def _tool_call_name(call: dict[str, Any]) -> str:
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    return str(function.get("name") or call.get("name") or "")


def _tool_call_arguments(call: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    arguments = function.get("arguments", call.get("arguments", {}))
    if isinstance(arguments, str):
        import json

        try:
            parsed = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            parsed = {"raw": arguments}
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    return arguments if isinstance(arguments, dict) else {}
