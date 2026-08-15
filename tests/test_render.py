from __future__ import annotations

from io import StringIO

from rich.console import Console

from paicli.entrypoints.repl import _bottom_toolbar, _prompt_message
from paicli.render import RichRenderer


def test_banner_renders_pi_home_layout():
    stream = StringIO()
    console = Console(file=stream, color_system=None, width=200)
    renderer = RichRenderer(console=console)

    renderer.banner(
        model="deepseek-v4-flash",
        provider="deepseek",
        cwd="/tmp/project",
        tools=12,
        version="0.1.0",
        api_key_configured=True,
        mcp_servers=1,
        skills=3,
        agents_files=2,
        hitl_mode="never",
    )

    output = stream.getvalue()
    assert "████████████" in output
    assert "  ██    ██" in output
    assert "SmartCLI v0.1.0" in output
    assert "Signed in API Key" in output
    assert "What's new (v0.1.0)" in output


def test_prompt_message_keeps_status_and_input_together():
    prompt = _prompt_message(
        cwd="/tmp/project",
        model="deepseek-v4-flash",
        tools=12,
        agents_files=2,
        mcp_servers=1,
        skills=3,
        stats={"total_tokens": 13187, "context_ratio": 0.013, "has_usage": True},
    )
    plain = "".join(text for _style, text in prompt)

    assert "2 AGENTS.md files" in plain
    assert "1 MCP server" in plain
    assert "3 skills · Tools 12" in plain
    assert "YOLO" not in plain
    assert "Shift+Tab" not in plain
    assert "deepseek-v4-flash" in plain
    assert "█░░░░░░░░░░░ 1%" in plain
    assert "/tmp/project" in plain
    assert "\n\n* " in plain
    assert plain.endswith("\n* ")


def test_bottom_toolbar_uses_runtime_summary_segments():
    toolbar = _bottom_toolbar(
        "/Users/me/project",
        "deepseek-v4-flash",
        {"turns": 1, "total_tokens": 13187, "context_ratio": 0.013, "has_usage": True},
    )

    assert ("class:toolbar.model", "deepseek-v4-flash") in toolbar
    assert ("class:toolbar.ctx.bar", "█░░░░░░░░░░░") in toolbar
    assert ("class:toolbar.ctx.value", "1%") in toolbar
    assert ("class:toolbar.cwd.value", "/Users/me/project") in toolbar
    assert not any(text == " TURN " for _style, text in toolbar)
    assert not any("Token" in text for _style, text in toolbar)


def test_text_deltas_render_as_markdown_on_turn_complete():
    stream = StringIO()
    console = Console(file=stream, color_system=None, width=120)
    renderer = RichRenderer(console=console, context_window=1000)

    renderer.handle({"type": "thinking_delta", "thinking": "需要先确认项目结构"})
    renderer.handle({"type": "text_delta", "text": "你好，我是 **Smart"})
    renderer.handle({"type": "text_delta", "text": "CLI**\n\n- `read_file`\n- **网页搜索**"})
    renderer.handle({"type": "usage", "usage": {"input_tokens": 250, "output_tokens": 50}})
    renderer.handle({"type": "turn_complete"})
    renderer.handle({"type": "done", "total_turns": 1, "total_tokens": 300})

    output = stream.getvalue()
    assert "Thinking" in output
    assert "需要先确认项目结构" in output
    assert "Final Output" in output
    assert "SmartCLI" in output
    assert "read_file" in output
    assert "网页搜索" in output
    assert "Run Summary" not in output
    assert "**SmartCLI**" not in output
    assert "`read_file`" not in output

    stats = renderer.toolbar_status()
    assert stats["turns"] == 1
    assert stats["input_tokens"] == 250
    assert stats["output_tokens"] == 50
    assert stats["total_tokens"] == 300
    assert stats["context_ratio"] == 0.25
    assert stats["has_usage"] is True


def test_context_usage_drives_toolbar_and_compression_is_concise():
    stream = StringIO()
    console = Console(file=stream, color_system=None, width=120)
    renderer = RichRenderer(console=console, context_window=1000)

    renderer.handle(
        {
            "type": "context_usage",
            "estimated_input_tokens": 500,
            "context_window": 1000,
            "output_reserve": 100,
            "tool_reserve": 50,
            "safety_margin": 50,
            "pressure_ratio": 0.7,
            "target_ratio": 0.55,
        }
    )
    renderer.handle(
        {
            "type": "context_compressed",
            "before_pressure": 0.84,
            "after_pressure": 0.56,
            "compacted_tool_results": 2,
            "summarized_messages": 6,
        }
    )
    renderer.handle({"type": "done", "total_turns": 1, "total_tokens": 0})

    stats = renderer.toolbar_status()
    assert stats["context_ratio"] == 0.7
    assert stats["estimated_input_tokens"] == 500
    assert stats["output_reserve"] == 100
    assert "Context compressed (automatic): 84% -> 56%" in stream.getvalue()


def test_interleaved_thinking_does_not_repeat_assistant_output_panels():
    stream = StringIO()
    console = Console(file=stream, color_system=None, width=120)
    renderer = RichRenderer(console=console)

    renderer.handle({"type": "text_delta", "text": "第一段"})
    renderer.handle({"type": "thinking_delta", "thinking": "中途补充思考"})
    renderer.handle({"type": "text_delta", "text": "第二段"})
    renderer.handle({"type": "turn_complete"})

    output = stream.getvalue()
    assert output.count("Assistant Output") == 0
    assert output.count("Final Output") == 1
    assert output.count("Thinking") == 1
    assert "第一段第二段" in output


def test_streaming_text_waits_for_turn_boundary_by_default():
    stream = StringIO()
    console = Console(file=stream, color_system=None, width=120, force_terminal=True)
    renderer = RichRenderer(console=console)

    renderer.handle({"type": "text_delta", "text": "chunk 1"})
    renderer.handle({"type": "text_delta", "text": "chunk 2"})

    assert "Assistant Output" not in stream.getvalue()
    renderer.handle({"type": "turn_complete"})
    assert stream.getvalue().count("Final Output") == 1


def test_tool_use_and_result_render_as_structured_panels():
    stream = StringIO()
    console = Console(file=stream, color_system=None, width=120)
    renderer = RichRenderer(console=console)

    renderer.handle({"type": "tool_call", "name": "list_dir", "input": {"path": "."}})
    renderer.handle(
        {
            "type": "tool_result",
            "name": "list_dir",
            "result": "README.md\nsrc/",
            "is_error": False,
        }
    )

    output = stream.getvalue()
    assert "Tool Use" in output
    assert "list_dir" in output
    assert '"path": "."' in output
    assert "Tool Result · list_dir · ok" in output
    assert "README.md" in output


def test_start_run_resets_token_usage():
    stream = StringIO()
    console = Console(file=stream, color_system=None, width=120)
    renderer = RichRenderer(console=console, context_window=1000)

    renderer.handle({"type": "usage", "usage": {"input_tokens": 900, "output_tokens": 10}})
    renderer.start_run()
    renderer.handle({"type": "usage", "usage": {"input_tokens": 100, "output_tokens": 20}})
    renderer.handle({"type": "done", "total_turns": 1, "total_tokens": 120})

    assert "900" not in stream.getvalue()
    stats = renderer.toolbar_status()
    assert stats["input_tokens"] == 100
    assert stats["output_tokens"] == 20
    assert stats["total_tokens"] == 120
    assert stats["context_ratio"] == 0.1


def test_missing_usage_keeps_toolbar_tokens_unavailable():
    stream = StringIO()
    console = Console(file=stream, color_system=None, width=120)
    renderer = RichRenderer(console=console, context_window=1000)

    renderer.handle({"type": "done", "total_turns": 1, "total_tokens": 0})

    assert "Run Summary" not in stream.getvalue()
    toolbar = _bottom_toolbar("/tmp/project", "deepseek-v4-flash", renderer.toolbar_status())
    assert ("class:toolbar.model", "deepseek-v4-flash") in toolbar
    assert ("class:toolbar.ctx.bar", "░░░░░░░░░░░░") in toolbar
    assert ("class:toolbar.ctx.value", "0%") in toolbar


def test_run_stopped_renders_the_guard_reason():
    stream = StringIO()
    console = Console(file=stream, color_system=None, width=120)
    renderer = RichRenderer(console=console)

    renderer.handle(
        {
            "type": "run_stopped",
            "reason": "repeated_tool_call",
            "message": "Repeated tool call detected.",
        }
    )

    output = stream.getvalue()
    assert "Agent Stopped" in output
    assert "repeated_tool_call" in output
    assert "Repeated tool call detected." in output


def test_stop_hook_and_budget_continuation_are_rendered():
    stream = StringIO()
    console = Console(file=stream, color_system=None, width=120)
    renderer = RichRenderer(console=console)

    renderer.handle(
        {
            "type": "stop_hook_review",
            "approved": False,
            "feedback": "Run tests before stopping.",
        }
    )
    renderer.handle(
        {
            "type": "model_redirected",
            "reason": "repeated_tool_call",
            "message": "Change parameters or approach.",
        }
    )
    renderer.handle(
        {
            "type": "budget_extended",
            "additional_turns": 20,
            "additional_tokens": 100000,
        }
    )

    output = stream.getvalue()
    assert "Stop Hook" in output
    assert "revision required" in output
    assert "Run tests before stopping." in output
    assert "Agent Correction" in output
    assert "Budget extended" in output
    assert "context preserved" in output


def test_connection_error_is_rendered_without_a_traceback():
    stream = StringIO()
    console = Console(file=stream, color_system=None, width=120)
    renderer = RichRenderer(console=console)

    renderer.handle(
        {
            "type": "error",
            "error": RuntimeError("raw internal error"),
            "message": "无法连接模型服务。当前上下文已保留。",
            "context_preserved": True,
        }
    )

    output = stream.getvalue()
    assert "Connection Error" in output
    assert "context preserved" in output
    assert "无法连接模型服务" in output
    assert "Traceback" not in output
    assert "raw internal error" not in output


def test_llm_retry_is_rendered_with_attempt_delay_and_reason():
    stream = StringIO()
    console = Console(file=stream, color_system=None, width=120)
    renderer = RichRenderer(console=console)

    renderer.handle(
        {
            "type": "llm_retry",
            "attempt": 1,
            "max_retries": 2,
            "delay_seconds": 0.5,
            "reason": "HTTP 429",
        }
    )

    output = stream.getvalue()
    assert "Model API Retry" in output
    assert "1/2" in output
    assert "HTTP 429" in output
    assert "0.5s" in output
