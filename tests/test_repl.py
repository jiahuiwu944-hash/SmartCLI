from __future__ import annotations

import asyncio
import io
from unittest.mock import MagicMock

import pytest
from rich.console import Console

from paicli.config import load_config
from paicli.entrypoints.repl import _handle_slash, _memory_command, _parse_search_args
from paicli.memory import MemoryManager


def _console() -> tuple[Console, io.StringIO]:
    stream = io.StringIO()
    return Console(file=stream, color_system=None, width=200), stream


class TestParseSearchArgs:
    def test_legacy_query_defaults_to_auto_mode(self):
        assert _parse_search_args("ToolExecutor") == ("auto", "ToolExecutor")

    def test_explicit_mode_flag(self):
        assert _parse_search_args("--mode symbol ToolExecutor") == ("symbol", "ToolExecutor")

    def test_all_supported_modes(self):
        for mode in ("auto", "symbol", "text", "references"):
            assert _parse_search_args(f"--mode {mode} some query") == (mode, "some query")

    def test_mode_flag_is_case_insensitive(self):
        assert _parse_search_args("--mode SYMBOL ToolExecutor") == ("symbol", "ToolExecutor")

    def test_mode_after_query_is_treated_as_query_text(self):
        assert _parse_search_args("ToolExecutor --mode symbol") == (
            "auto",
            "ToolExecutor --mode symbol",
        )

    def test_missing_query_raises(self):
        with pytest.raises(ValueError, match="missing search query"):
            _parse_search_args("")

    def test_mode_without_query_raises(self):
        with pytest.raises(ValueError, match="missing search query"):
            _parse_search_args("--mode symbol")

    def test_missing_mode_value_raises(self):
        with pytest.raises(ValueError, match="missing value for --mode"):
            _parse_search_args("--mode")

    def test_unsupported_mode_raises(self):
        with pytest.raises(ValueError, match="unsupported search mode: semantic"):
            _parse_search_args("--mode semantic ToolExecutor")


def _run_slash(
    raw: str,
    tmp_path,
    monkeypatch,
) -> str:
    """Run _handle_slash for a /search command and return console output."""
    captured_calls: list[tuple] = []

    class _FakeNavigator:
        def __init__(self, cwd):
            self.cwd = cwd

        def search(self, query, *, mode="auto", limit=20):
            captured_calls.append((query, mode, limit))
            return [], False

    monkeypatch.setattr("paicli.entrypoints.repl.CodeNavigator", _FakeNavigator)
    console, stream = _console()
    config = load_config(project_root=tmp_path)
    agent = MagicMock()
    registry = MagicMock()
    asyncio.run(_handle_slash(raw, console, str(tmp_path), config, agent, registry))
    return stream.getvalue(), captured_calls


def test_slash_search_legacy_query_passes_auto_mode(tmp_path, monkeypatch):
    output, calls = _run_slash("/search ToolExecutor", tmp_path, monkeypatch)

    assert calls == [("ToolExecutor", "auto", 20)]
    assert "(no matches)" in output


def test_slash_search_explicit_mode_is_forwarded(tmp_path, monkeypatch):
    output, calls = _run_slash("/search --mode symbol ToolExecutor", tmp_path, monkeypatch)

    assert calls == [("ToolExecutor", "symbol", 20)]
    assert "(no matches)" in output


def test_slash_search_without_query_shows_usage(tmp_path, monkeypatch):
    output, calls = _run_slash("/search", tmp_path, monkeypatch)

    assert calls == []
    assert "Usage:" in output
    assert "--mode auto|symbol|text|references" in output


def test_slash_search_with_invalid_mode_shows_usage(tmp_path, monkeypatch):
    output, calls = _run_slash("/search --mode semantic ToolExecutor", tmp_path, monkeypatch)

    assert calls == []
    assert "unsupported search mode: semantic" in output
    assert "Usage:" in output


def test_memory_history_audit_and_restore_commands(tmp_path):
    config = load_config(project_root=tmp_path)
    config.memory.long_term_db_path = str(tmp_path / "memory.db")
    manager = MemoryManager(config.memory.long_term_db_path, scope=tmp_path)
    old_id = manager.save("Uses MySQL", memory_key="project.database")
    manager.save("Uses PostgreSQL", memory_key="project.database")

    console, stream = _console()
    asyncio.run(_memory_command("history", console, str(tmp_path), config))
    assert "superseded" in stream.getvalue()
    assert "Uses PostgreSQL" in stream.getvalue()

    console, stream = _console()
    asyncio.run(_memory_command("audit", console, str(tmp_path), config))
    assert "inserted" in stream.getvalue()
    assert "superseded" in stream.getvalue()

    console, stream = _console()
    asyncio.run(_memory_command(f"restore {old_id}", console, str(tmp_path), config))
    assert f"Restored memory #{old_id}" in stream.getvalue()
    assert manager.list()[0].content == "Uses MySQL"
