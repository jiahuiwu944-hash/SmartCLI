from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack, suppress
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from pydantic import AnyUrl

from paicli.mcp.config import McpServerSpec, load_mcp_server_specs
from paicli.tools.base import Tool, ToolContext, ToolResult, object_schema


@dataclass(slots=True)
class _SessionRequest:
    operation: str
    arguments: tuple[Any, ...]
    future: asyncio.Future[Any]


class _PersistentMcpConnection:
    """Own one MCP session in one task and reconnect it after transport failures."""

    def __init__(self, spec: McpServerSpec, project_root: str):
        self.spec = spec
        self.project_root = project_root
        self.capabilities: Any = None
        self.server_info: Any = None
        self._queue: asyncio.Queue[_SessionRequest | None] = asyncio.Queue()
        self._ready: asyncio.Future[None] | None = None
        self._runner: asyncio.Task[None] | None = None
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None

    async def start(self) -> None:
        if self._runner and not self._runner.done():
            return
        loop = asyncio.get_running_loop()
        self._ready = loop.create_future()
        self._runner = asyncio.create_task(
            self._run(),
            name=f"smartcli-mcp-{self.spec.name}",
        )
        await asyncio.wait_for(asyncio.shield(self._ready), timeout=self.spec.timeout)

    async def request(self, operation: str, *arguments: Any) -> Any:
        if not self._runner or self._runner.done():
            await self.start()
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        await self._queue.put(_SessionRequest(operation, arguments, future))
        return await future

    async def close(self) -> None:
        runner = self._runner
        if not runner:
            return
        if not runner.done():
            await self._queue.put(None)
        with suppress(asyncio.CancelledError, Exception):
            await runner
        self._runner = None

    async def _run(self) -> None:
        try:
            await self._connect()
        except Exception as exc:  # noqa: BLE001 - surfaced through the ready future
            if self._ready and not self._ready.done():
                self._ready.set_exception(exc)
            await self._disconnect()
            return
        if self._ready and not self._ready.done():
            self._ready.set_result(None)

        try:
            while True:
                request = await self._queue.get()
                if request is None:
                    return
                await self._execute_with_reconnect(request)
        finally:
            await self._disconnect()
            error = RuntimeError(f"MCP connection {self.spec.name} is closed")
            while not self._queue.empty():
                request = self._queue.get_nowait()
                if request is not None and not request.future.done():
                    request.future.set_exception(error)

    async def _execute_with_reconnect(self, request: _SessionRequest) -> None:
        attempts = max(0, int(self.spec.max_retries)) + 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                if self._session is None:
                    await self._connect()
                result = await asyncio.wait_for(
                    self._dispatch(request.operation, request.arguments),
                    timeout=self.spec.timeout,
                )
            except Exception as exc:  # noqa: BLE001 - transport boundary
                last_error = exc
                await self._disconnect()
                if attempt + 1 < attempts:
                    await asyncio.sleep(self.spec.retry_base_delay * (2**attempt))
                    continue
                break
            if not request.future.done():
                request.future.set_result(result)
            return
        if not request.future.done():
            request.future.set_exception(
                last_error or RuntimeError(f"MCP request failed: {request.operation}")
            )

    async def _connect(self) -> None:
        stack = AsyncExitStack()
        try:
            if self.spec.type in {"stdio", "local"}:
                if not self.spec.command:
                    raise ValueError(f"MCP server {self.spec.name} is missing command")
                params = StdioServerParameters(
                    command=_stdio_command(self.spec.command),
                    args=self.spec.args,
                    env={**os.environ, **self.spec.env},
                    cwd=self.spec.cwd or self.project_root,
                )
                errlog = stack.enter_context(
                    Path(os.devnull).open("w", encoding="utf-8")  # noqa: SIM115
                )
                read, write = await stack.enter_async_context(stdio_client(params, errlog=errlog))
            elif self.spec.type in {"http", "streamable_http", "streamable-http"}:
                if not self.spec.url:
                    raise ValueError(f"MCP server {self.spec.name} is missing url")
                read, write, _session_id = await stack.enter_async_context(
                    streamablehttp_client(
                        self.spec.url,
                        headers=self.spec.headers or None,
                        timeout=self.spec.timeout,
                    )
                )
            else:
                raise ValueError(f"Unsupported MCP transport: {self.spec.type}")
            session = await stack.enter_async_context(ClientSession(read, write))
            initialized = await asyncio.wait_for(session.initialize(), timeout=self.spec.timeout)
        except BaseException:
            await stack.aclose()
            raise
        self._stack = stack
        self._session = session
        self.capabilities = initialized.capabilities
        self.server_info = initialized.serverInfo

    async def _disconnect(self) -> None:
        stack, self._stack = self._stack, None
        self._session = None
        if stack:
            with suppress(Exception):
                await stack.aclose()

    async def _dispatch(self, operation: str, arguments: tuple[Any, ...]) -> Any:
        session = self._session
        if session is None:
            raise RuntimeError(f"MCP server {self.spec.name} is not connected")
        if operation == "list_tools":
            return await _list_all_pages(session.list_tools, "tools")
        if operation == "call_tool":
            name, payload = arguments
            return await session.call_tool(
                name,
                payload,
                read_timeout_seconds=timedelta(seconds=self.spec.timeout),
            )
        if operation == "list_resources":
            return await _list_all_pages(session.list_resources, "resources")
        if operation == "read_resource":
            return await session.read_resource(AnyUrl(str(arguments[0])))
        if operation == "list_prompts":
            return await _list_all_pages(session.list_prompts, "prompts")
        if operation == "get_prompt":
            name, payload = arguments
            return await session.get_prompt(name, payload)
        raise ValueError(f"Unsupported MCP operation: {operation}")


class McpClientManager:
    def __init__(self, project_root: str | Path):
        self.project_root = str(Path(project_root).resolve())
        self.specs = load_mcp_server_specs(self.project_root)
        self.last_errors: dict[str, str] = {}
        self._connections: dict[str, _PersistentMcpConnection] = {}

    @property
    def connected_servers(self) -> tuple[str, ...]:
        return tuple(self._connections)

    async def load_tools(self) -> list[Tool]:
        await self.close()
        self.last_errors.clear()
        specs = [spec for spec in self.specs.values() if spec.enabled]
        results = await asyncio.gather(
            *(self._load_server(spec) for spec in specs),
            return_exceptions=True,
        )
        tools: list[Tool] = []
        for spec, result in zip(specs, results, strict=True):
            if isinstance(result, BaseException):
                self.last_errors[spec.name] = str(result)
            else:
                tools.extend(result)
        return tools

    async def close(self) -> None:
        connections = list(self._connections.values())
        self._connections.clear()
        if connections:
            await asyncio.gather(*(item.close() for item in connections), return_exceptions=True)

    async def _load_server(self, spec: McpServerSpec) -> list[Tool]:
        connection = _PersistentMcpConnection(spec, self.project_root)
        try:
            await connection.start()
            self._connections[spec.name] = connection
            tools: list[Tool] = []
            capabilities = connection.capabilities
            if getattr(capabilities, "tools", None) is not None:
                tools.extend(await self._tools_for_server(spec))
            if getattr(capabilities, "resources", None) is not None:
                tools.extend(self._virtual_resource_tools(spec))
            if getattr(capabilities, "prompts", None) is not None:
                tools.extend(self._virtual_prompt_tools(spec))
            return tools
        except BaseException:
            self._connections.pop(spec.name, None)
            await connection.close()
            raise

    def _connection(self, spec: McpServerSpec) -> _PersistentMcpConnection:
        connection = self._connections.get(spec.name)
        if connection is None:
            raise RuntimeError(f"MCP server {spec.name} is not connected")
        return connection

    async def list_server_tools(self, spec: McpServerSpec) -> list[Any]:
        return list(await self._connection(spec).request("list_tools"))

    async def call_server_tool(
        self,
        spec: McpServerSpec,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        result = await self._connection(spec).request("call_tool", tool_name, arguments)
        content = _content_to_text(result.content)
        return ToolResult(content=content, is_error=bool(result.isError))

    async def list_resources(self, spec: McpServerSpec) -> ToolResult:
        resources = await self._connection(spec).request("list_resources")
        lines = [
            f"{resource.uri} {resource.name or ''} {resource.description or ''}".strip()
            for resource in resources
        ]
        return ToolResult("\n".join(lines) or "(no resources)")

    async def read_resource(self, spec: McpServerSpec, uri: str) -> ToolResult:
        result = await self._connection(spec).request("read_resource", uri)
        return ToolResult(_content_to_text(result.contents))

    async def list_prompts(self, spec: McpServerSpec) -> ToolResult:
        prompts = await self._connection(spec).request("list_prompts")
        lines = [f"{prompt.name} {prompt.description or ''}".strip() for prompt in prompts]
        return ToolResult("\n".join(lines) or "(no prompts)")

    async def get_prompt(
        self,
        spec: McpServerSpec,
        name: str,
        arguments: dict[str, str] | None = None,
    ) -> ToolResult:
        result = await self._connection(spec).request("get_prompt", name, arguments or {})
        return ToolResult(_content_to_text(result.messages))

    async def _tools_for_server(self, spec: McpServerSpec) -> list[Tool]:
        remote_tools = await self.list_server_tools(spec)
        wrapped: list[Tool] = []
        for remote_tool in remote_tools:
            tool_name = str(remote_tool.name)
            local_name = f"mcp__{spec.name}__{tool_name}"
            schema = remote_tool.inputSchema or object_schema({})
            annotations = getattr(remote_tool, "annotations", None)
            read_only = bool(getattr(annotations, "readOnlyHint", False))

            async def handler(
                payload: dict[str, Any],
                context: ToolContext,
                *,
                server_spec: McpServerSpec = spec,
                remote_name: str = tool_name,
            ) -> ToolResult:
                _ = context
                return await self.call_server_tool(server_spec, remote_name, payload)

            wrapped.append(
                Tool(
                    name=local_name,
                    description=remote_tool.description or f"MCP tool {tool_name}",
                    parameters=schema,
                    handler=handler,
                    is_read_only=read_only,
                    is_concurrency_safe=False,
                    danger_level="safe" if read_only else "medium",
                    requires_approval=not read_only,
                )
            )
        return wrapped

    def _virtual_resource_tools(self, spec: McpServerSpec) -> list[Tool]:
        async def list_handler(payload: dict[str, Any], context: ToolContext) -> ToolResult:
            _ = payload, context
            return await self.list_resources(spec)

        async def read_handler(payload: dict[str, Any], context: ToolContext) -> ToolResult:
            _ = context
            return await self.read_resource(spec, str(payload["uri"]))

        return [
            Tool(
                name=f"mcp__{spec.name}__list_resources",
                description=f"List MCP resources from {spec.name}.",
                parameters=object_schema({}),
                handler=list_handler,
                is_read_only=True,
            ),
            Tool(
                name=f"mcp__{spec.name}__read_resource",
                description=f"Read an MCP resource from {spec.name}.",
                parameters=object_schema(
                    {"uri": {"type": "string", "description": "Resource URI"}},
                    ["uri"],
                ),
                required_keys=["uri"],
                handler=read_handler,
                is_read_only=True,
            ),
        ]

    def _virtual_prompt_tools(self, spec: McpServerSpec) -> list[Tool]:
        async def list_handler(payload: dict[str, Any], context: ToolContext) -> ToolResult:
            _ = payload, context
            return await self.list_prompts(spec)

        async def get_handler(payload: dict[str, Any], context: ToolContext) -> ToolResult:
            _ = context
            arguments = payload.get("arguments")
            if arguments is not None and not isinstance(arguments, dict):
                return ToolResult("arguments must be an object", is_error=True)
            return await self.get_prompt(
                spec,
                str(payload["name"]),
                {str(k): str(v) for k, v in (arguments or {}).items()},
            )

        return [
            Tool(
                name=f"mcp__{spec.name}__list_prompts",
                description=f"List MCP prompts from {spec.name}.",
                parameters=object_schema({}),
                handler=list_handler,
                is_read_only=True,
            ),
            Tool(
                name=f"mcp__{spec.name}__get_prompt",
                description=f"Get an MCP prompt from {spec.name}.",
                parameters=object_schema(
                    {
                        "name": {"type": "string", "description": "Prompt name"},
                        "arguments": {"type": "object", "description": "Prompt arguments"},
                    },
                    ["name"],
                ),
                required_keys=["name"],
                handler=get_handler,
                is_read_only=True,
            ),
        ]


async def _list_all_pages(method, attribute: str) -> list[Any]:
    items: list[Any] = []
    cursor: str | None = None
    while True:
        result = await method(cursor=cursor)
        items.extend(getattr(result, attribute))
        cursor = getattr(result, "nextCursor", None)
        if not cursor:
            return items


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(filter(None, (_content_to_text(item) for item in content)))
    if hasattr(content, "text"):
        return str(content.text)
    if hasattr(content, "data") and hasattr(content, "mimeType"):
        data = str(content.data)
        return f"[image {content.mimeType} base64 chars={len(data)}]"
    if hasattr(content, "resource"):
        return _content_to_text(content.resource)
    if hasattr(content, "model_dump"):
        return json.dumps(content.model_dump(mode="json"), ensure_ascii=False)
    return str(content)


def _stdio_command(command: str) -> str:
    """Keep Python MCP servers in SmartCLI's active virtual environment."""
    if command.lower() in {"python", "python3", "python.exe"}:
        return sys.executable
    return command
