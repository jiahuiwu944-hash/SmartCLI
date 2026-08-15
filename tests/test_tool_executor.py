from __future__ import annotations

import asyncio
import json

from paicli.config import load_config
from paicli.tools.base import Tool, ToolContext, ToolResult, object_schema
from paicli.tools.executor import ToolExecutor
from paicli.tools.registry import ToolRegistry


def _call(name: str, call_id: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps({})},
    }


def _context(tmp_path, monkeypatch, *, max_concurrent_read: int = 4) -> ToolContext:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config = load_config(project_root=tmp_path)
    config.tools.max_concurrent_read = max_concurrent_read
    return ToolContext(cwd=str(tmp_path), config=config)


def _tool(name: str, handler, *, read_only: bool = True) -> Tool:
    return Tool(
        name=name,
        description=name,
        parameters=object_schema({}),
        handler=handler,
        is_read_only=read_only,
        is_concurrency_safe=read_only,
    )


def test_adjacent_safe_reads_run_concurrently_and_keep_result_order(tmp_path, monkeypatch):
    started: list[str] = []
    both_started = asyncio.Event()
    release = asyncio.Event()

    async def read_handler(payload, context):  # noqa: ARG001
        name = asyncio.current_task().get_name()
        started.append(name)
        if len(started) == 2:
            both_started.set()
        await asyncio.wait_for(release.wait(), timeout=1)
        return ToolResult(content=name)

    async def run():
        registry = ToolRegistry()
        registry.register(_tool("read_a", read_handler))
        registry.register(_tool("read_b", read_handler))
        executor = ToolExecutor(registry)
        task = asyncio.create_task(
            executor.execute_all(
                [_call("read_a", "call_a"), _call("read_b", "call_b")],
                _context(tmp_path, monkeypatch),
            )
        )
        await asyncio.wait_for(both_started.wait(), timeout=1)
        release.set()
        return await task

    results = asyncio.run(run())

    assert len(started) == 2
    assert [result.tool_use_id for result in results] == ["call_a", "call_b"]


def test_side_effecting_tool_splits_read_batches_without_reordering(tmp_path, monkeypatch):
    events: list[str] = []
    first_batch_started = asyncio.Event()
    release_first_batch = asyncio.Event()

    async def first_read(payload, context):  # noqa: ARG001
        events.append("read_a:start")
        first_batch_started.set()
        await asyncio.wait_for(release_first_batch.wait(), timeout=1)
        events.append("read_a:end")
        return ToolResult(content="read_a")

    async def write_handler(payload, context):  # noqa: ARG001
        events.append("write")
        return ToolResult(content="write")

    async def second_read(payload, context):  # noqa: ARG001
        events.append("read_b")
        return ToolResult(content="read_b")

    async def run():
        registry = ToolRegistry()
        registry.register(_tool("read_a", first_read))
        registry.register(_tool("write", write_handler, read_only=False))
        registry.register(_tool("read_b", second_read))
        executor = ToolExecutor(registry)
        task = asyncio.create_task(
            executor.execute_all(
                [
                    _call("read_a", "call_a"),
                    _call("write", "call_write"),
                    _call("read_b", "call_b"),
                ],
                _context(tmp_path, monkeypatch),
            )
        )
        await asyncio.wait_for(first_batch_started.wait(), timeout=1)
        assert events == ["read_a:start"]
        release_first_batch.set()
        return await task

    results = asyncio.run(run())

    assert events == ["read_a:start", "read_a:end", "write", "read_b"]
    assert [result.tool_use_id for result in results] == [
        "call_a",
        "call_write",
        "call_b",
    ]


def test_write_before_read_is_not_reordered(tmp_path, monkeypatch):
    events: list[str] = []

    async def write_handler(payload, context):  # noqa: ARG001
        events.append("write")
        return ToolResult(content="write")

    async def read_handler(payload, context):  # noqa: ARG001
        events.append("read")
        return ToolResult(content="read")

    registry = ToolRegistry()
    registry.register(_tool("write", write_handler, read_only=False))
    registry.register(_tool("read", read_handler))

    results = asyncio.run(
        ToolExecutor(registry).execute_all(
            [_call("write", "call_write"), _call("read", "call_read")],
            _context(tmp_path, monkeypatch),
        )
    )

    assert events == ["write", "read"]
    assert [result.tool_use_id for result in results] == ["call_write", "call_read"]


def test_read_concurrency_respects_configured_limit(tmp_path, monkeypatch):
    active = 0
    peak = 0

    async def read_handler(payload, context):  # noqa: ARG001
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return ToolResult(content="ok")

    registry = ToolRegistry()
    registry.register(_tool("read", read_handler))

    results = asyncio.run(
        ToolExecutor(registry).execute_all(
            [_call("read", f"call_{index}") for index in range(5)],
            _context(tmp_path, monkeypatch, max_concurrent_read=2),
        )
    )

    assert peak == 2
    assert [result.tool_use_id for result in results] == [
        "call_0",
        "call_1",
        "call_2",
        "call_3",
        "call_4",
    ]
