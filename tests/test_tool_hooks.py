from __future__ import annotations

import asyncio
import json

from paicli.agent import QueryEngine
from paicli.config import load_config
from paicli.tools import get_builtin_tools
from paicli.tools.base import Tool, ToolContext, ToolResult, object_schema
from paicli.tools.executor import ToolExecutor
from paicli.tools.hooks import (
    PreToolDecision,
    ToolHookContext,
    ToolHookManager,
    ToolLifecycleHook,
    cleanup_managed_scratch,
    default_tool_hooks,
)
from paicli.tools.registry import ToolRegistry


def _call(name: str, arguments: dict, call_id: str = "call_1") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _context(tmp_path, monkeypatch, *, approval_callback=None):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config = load_config(project_root=tmp_path)
    return ToolContext(
        cwd=str(tmp_path),
        config=config,
        approval_callback=approval_callback,
    )


def test_pre_and_post_hooks_modify_input_and_result(tmp_path, monkeypatch):
    events: list[str] = []

    async def handler(payload, context):  # noqa: ARG001
        events.append(f"handler:{payload['value']}")
        return ToolResult(content=str(payload["value"]))

    class RewriteHook(ToolLifecycleHook):
        async def before_tool(self, context):
            events.append("before")
            return PreToolDecision(updated_input={"value": context.input_data["value"] + 1})

        async def after_tool(self, context, result):  # noqa: ARG002
            events.append("after")
            return ToolResult(content=result.content + ":post")

    tool = Tool(
        name="increment",
        description="increment",
        parameters=object_schema({"value": {"type": "integer"}}, ["value"]),
        handler=handler,
        required_keys=["value"],
    )
    registry = ToolRegistry()
    registry.register(tool)
    executor = ToolExecutor(registry, ToolHookManager([RewriteHook()]))

    results = asyncio.run(
        executor.execute_all([_call("increment", {"value": 1})], _context(tmp_path, monkeypatch))
    )

    assert events == ["before", "handler:2", "after"]
    assert results[0].content == "2:post"
    assert results[0].tool_use_id == "call_1"


def test_pre_hook_can_deny_without_executing_tool(tmp_path, monkeypatch):
    executed = False

    async def handler(payload, context):  # noqa: ARG001
        nonlocal executed
        executed = True
        return ToolResult(content="unexpected")

    class DenyHook(ToolLifecycleHook):
        async def before_tool(self, context):  # noqa: ARG002
            return PreToolDecision(behavior="deny", message="blocked by custom policy")

    tool = Tool(
        name="dangerous",
        description="dangerous",
        parameters=object_schema({}),
        handler=handler,
        is_read_only=False,
    )
    registry = ToolRegistry()
    registry.register(tool)

    results = asyncio.run(
        ToolExecutor(registry, ToolHookManager([DenyHook()])).execute_all(
            [_call("dangerous", {})],
            _context(tmp_path, monkeypatch),
        )
    )

    assert executed is False
    assert results[0].is_error is True
    assert results[0].content == "blocked by custom policy"


def test_error_hook_can_convert_exception_to_model_feedback(tmp_path, monkeypatch):
    async def handler(payload, context):  # noqa: ARG001
        raise ValueError("bad input")

    class RecoverHook(ToolLifecycleHook):
        async def on_tool_error(self, context: ToolHookContext, error: Exception):
            return ToolResult(
                content=f"recoverable:{context.tool_name}:{error}",
                is_error=True,
            )

    tool = Tool(
        name="unstable",
        description="unstable",
        parameters=object_schema({}),
        handler=handler,
    )
    registry = ToolRegistry()
    registry.register(tool)

    results = asyncio.run(
        ToolExecutor(registry, ToolHookManager([RecoverHook()])).execute_all(
            [_call("unstable", {})],
            _context(tmp_path, monkeypatch),
        )
    )

    assert results[0].content == "recoverable:unstable:bad input"
    assert results[0].is_error is True
    assert results[0].tool_use_id == "call_1"


def test_default_approval_hook_preserves_hitl_behavior(tmp_path, monkeypatch):
    executed = False

    async def handler(payload, context):  # noqa: ARG001
        nonlocal executed
        executed = True
        return ToolResult(content="unexpected")

    tool = Tool(
        name="write_demo",
        description="write demo",
        parameters=object_schema({}),
        handler=handler,
        is_read_only=False,
        requires_approval=True,
    )
    registry = ToolRegistry()
    registry.register(tool)
    context = _context(tmp_path, monkeypatch, approval_callback=lambda request: "deny")

    results = asyncio.run(ToolExecutor(registry).execute_all([_call("write_demo", {})], context))

    assert executed is False
    assert results[0].is_error is True
    assert "denied by approval policy" in results[0].content


class HookAwareClient:
    model_name = "fake-model"
    provider_name = "fake-provider"
    max_context_window = 1000

    def __init__(self):
        self.calls = 0

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        self.calls += 1
        if self.calls == 1:
            yield {
                "type": "tool_call_delta",
                "tool_call": {
                    "index": 0,
                    "id": "call_echo",
                    "function": {"name": "echo", "arguments": '{"value":"hello"}'},
                },
            }
            yield {"type": "message_end", "stop_reason": "tool_use"}
            return
        assert messages[-1].role == "tool"
        assert messages[-1].content == "hello:hooked"
        yield {"type": "text_delta", "text": "done"}
        yield {"type": "message_end", "stop_reason": "end_turn"}


def test_query_engine_accepts_a_custom_tool_hook_manager(tmp_path, monkeypatch):
    async def handler(payload, context):  # noqa: ARG001
        return ToolResult(content=payload["value"])

    class AppendHook(ToolLifecycleHook):
        async def after_tool(self, context, result):  # noqa: ARG002
            return ToolResult(content=result.content + ":hooked")

    tool = Tool(
        name="echo",
        description="echo",
        parameters=object_schema({"value": {"type": "string"}}, ["value"]),
        handler=handler,
        required_keys=["value"],
    )
    registry = ToolRegistry()
    registry.register(tool)
    config = _context(tmp_path, monkeypatch).config
    config.agent.stop_hook_enabled = False
    engine = QueryEngine(
        llm_client=HookAwareClient(),
        tool_registry=registry,
        config=config,
        cwd=str(tmp_path),
        tool_hook_manager=ToolHookManager([AppendHook()]),
    )

    result = asyncio.run(engine.ask_complete_async("echo hello"))

    assert result.text == "done"
    assert result.completed is True


def test_input_rewrite_runs_before_default_approval_hook(tmp_path, monkeypatch):
    approved_inputs = []
    executed_inputs = []

    async def handler(payload, context):  # noqa: ARG001
        executed_inputs.append(payload)
        return ToolResult(content="ok")

    class NormalizeHook(ToolLifecycleHook):
        async def before_tool(self, context):
            return PreToolDecision(updated_input={"value": context.input_data["value"].strip()})

    tool = Tool(
        name="normalize",
        description="normalize",
        parameters=object_schema({"value": {"type": "string"}}, ["value"]),
        handler=handler,
        is_read_only=False,
        requires_approval=True,
        required_keys=["value"],
    )
    registry = ToolRegistry()
    registry.register(tool)
    hooks = default_tool_hooks()
    hooks.register(NormalizeHook())

    def approve(request):
        approved_inputs.append(request["input"])
        return "approve"

    results = asyncio.run(
        ToolExecutor(registry, hooks).execute_all(
            [_call("normalize", {"value": " hello "})],
            _context(tmp_path, monkeypatch, approval_callback=approve),
        )
    )

    assert results[0].is_error is False
    assert approved_inputs == [{"value": "hello"}]
    assert executed_inputs == [{"value": "hello"}]


def test_shell_guard_blocks_source_mutating_patch_script(tmp_path, monkeypatch):
    executed = False
    patch_script = tmp_path / "tmp_patch_source.py"
    patch_script.write_text(
        "path = 'src/demo.py'\n"
        "text = open(path, encoding='utf-8').read()\n"
        "open(path, 'w', encoding='utf-8').write(text.replace('a', 'b'))\n",
        encoding="utf-8",
    )

    async def handler(payload, context):  # noqa: ARG001
        nonlocal executed
        executed = True
        return ToolResult(content="unexpected")

    tool = Tool(
        name="bash",
        description="shell",
        parameters=object_schema({"command": {"type": "string"}}, ["command"]),
        handler=handler,
        required_keys=["command"],
        is_read_only=False,
    )
    registry = ToolRegistry()
    registry.register(tool)
    context = _context(tmp_path, monkeypatch)
    context.config.policy.hitl_mode = "never"

    result = asyncio.run(
        ToolExecutor(registry).execute_all(
            [_call("bash", {"command": "python tmp_patch_source.py"})], context
        )
    )[0]

    assert executed is False
    assert result.is_error is True
    assert "source-write guard" in result.content
    assert "write_file" in result.content


def test_shell_guard_allows_tests_and_diagnostics(tmp_path, monkeypatch):
    executed = False
    test_file = tmp_path / "tests" / "test_demo.py"
    test_file.parent.mkdir()
    test_file.write_text(
        "from pathlib import Path\n"
        "def test_write_fixture():\n"
        "    Path('src/demo.py').write_text('fixture')\n",
        encoding="utf-8",
    )

    async def handler(payload, context):  # noqa: ARG001
        nonlocal executed
        executed = True
        return ToolResult(content="tests passed")

    tool = Tool(
        name="bash",
        description="shell",
        parameters=object_schema({"command": {"type": "string"}}, ["command"]),
        handler=handler,
        required_keys=["command"],
        is_read_only=False,
    )
    registry = ToolRegistry()
    registry.register(tool)
    context = _context(tmp_path, monkeypatch)
    context.config.policy.hitl_mode = "never"

    result = asyncio.run(
        ToolExecutor(registry).execute_all(
            [_call("bash", {"command": "python -m pytest tests/test_demo.py -q"})], context
        )
    )[0]

    assert executed is True
    assert result.is_error is False


def test_shell_guard_checks_only_direct_script_in_chained_pytest_command(
    tmp_path, monkeypatch
):
    executed = False
    probe = tmp_path / ".paicli" / "tmp" / "verify.py"
    probe.parent.mkdir(parents=True)
    probe.write_text("print('verified')\n", encoding="utf-8")
    test_file = tmp_path / "tests" / "test_render.py"
    test_file.parent.mkdir()
    test_file.write_text(
        "from pathlib import Path\n"
        "def test_renderer(tmp_path):\n"
        "    (tmp_path / 'render.py').write_text('fixture')\n",
        encoding="utf-8",
    )

    async def handler(payload, context):  # noqa: ARG001
        nonlocal executed
        executed = True
        return ToolResult(content="verified and tested")

    tool = Tool(
        name="bash",
        description="shell",
        parameters=object_schema({"command": {"type": "string"}}, ["command"]),
        handler=handler,
        required_keys=["command"],
        is_read_only=False,
    )
    registry = ToolRegistry()
    registry.register(tool)
    context = _context(tmp_path, monkeypatch)
    context.config.policy.hitl_mode = "never"

    result = asyncio.run(
        ToolExecutor(registry).execute_all(
            [
                _call(
                    "bash",
                    {
                        "command": (
                            "python .paicli/tmp/verify.py && "
                            "python -m pytest tests/test_render.py -q"
                        )
                    },
                )
            ],
            context,
        )
    )[0]

    assert executed is True
    assert result.is_error is False


def test_scratch_files_are_isolated_tracked_and_cleaned(tmp_path, monkeypatch):
    registry = ToolRegistry()
    registry.register_all(get_builtin_tools())
    context = _context(tmp_path, monkeypatch)
    context.config.policy.hitl_mode = "never"

    denied = asyncio.run(
        ToolExecutor(registry).execute_all(
            [_call("write_file", {"path": "tmp_probe.py", "content": "print('ok')\n"})],
            context,
        )
    )[0]
    allowed = asyncio.run(
        ToolExecutor(registry).execute_all(
            [
                _call(
                    "write_file",
                    {"path": ".paicli/tmp/tmp_probe.py", "content": "print('ok')\n"},
                    "call_2",
                )
            ],
            context,
        )
    )[0]

    managed = tmp_path / ".paicli" / "tmp" / "tmp_probe.py"
    assert denied.is_error is True
    assert ".paicli/tmp/tmp_probe.py" in denied.content
    assert allowed.is_error is False
    assert managed.exists()
    assert managed.resolve() in context.scratch_files
    assert cleanup_managed_scratch(context) == [str(managed.relative_to(tmp_path))]
    assert not managed.exists()


def test_exploration_guard_injects_convergence_feedback(tmp_path, monkeypatch):
    async def handler(payload, context):  # noqa: ARG001
        return ToolResult(content="diagnostic result")

    tool = Tool(
        name="bash",
        description="shell",
        parameters=object_schema({"command": {"type": "string"}}, ["command"]),
        handler=handler,
        required_keys=["command"],
        is_read_only=False,
    )
    registry = ToolRegistry()
    registry.register(tool)
    context = _context(tmp_path, monkeypatch)
    context.config.policy.hitl_mode = "never"
    context.config.agent.exploration_tool_call_limit = 2

    results = asyncio.run(
        ToolExecutor(registry).execute_all(
            [
                _call("bash", {"command": "echo first"}, "call_1"),
                _call("bash", {"command": "echo second"}, "call_2"),
            ],
            context,
        )
    )

    assert "exploration guard" not in results[0].content
    assert "exploration guard" in results[1].content
    assert "converge" in results[1].content
