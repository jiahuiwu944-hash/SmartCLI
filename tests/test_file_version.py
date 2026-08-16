from __future__ import annotations

import asyncio
import json

import pytest

from paicli.agent import QueryEngine
from paicli.codeintel import ContextLedger
from paicli.config import load_config
from paicli.policy import AuditLog
from paicli.tools import ToolRegistry, get_builtin_tools
from paicli.tools.base import ToolContext
from paicli.tools.builtins import read_file, write_file
from paicli.tools.executor import ToolExecutor
from paicli.tools.file_version import content_version


def _context(tmp_path, monkeypatch, *, mode: str = "warn") -> ToolContext:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config = load_config(project_root=tmp_path)
    config.tools.file_version_check = mode
    return ToolContext(cwd=str(tmp_path), config=config)


def _json(result) -> dict:
    return json.loads(result.content)


def test_read_file_returns_full_file_version_for_partial_read(tmp_path, monkeypatch):
    path = tmp_path / "note.txt"
    raw = b"one\ntwo\nthree\n"
    path.write_bytes(raw)

    result = asyncio.run(
        read_file(
            {"path": "note.txt", "offset": 2, "limit": 1},
            _context(tmp_path, monkeypatch),
        )
    )

    assert f"version: {content_version(raw)}" in result.content
    assert "2: two" in result.content
    assert "1: one" not in result.content


def test_matching_version_allows_atomic_overwrite(tmp_path, monkeypatch):
    path = tmp_path / "note.txt"
    path.write_text("old", encoding="utf-8")
    version = content_version(b"old")

    result = asyncio.run(
        write_file(
            {"path": "note.txt", "content": "new", "expected_version": version},
            _context(tmp_path, monkeypatch),
        )
    )

    data = _json(result)
    assert result.is_error is False
    assert data["status"] == "WRITE_OK"
    assert data["previous_version"] == version
    assert data["version"] == content_version(b"new")
    assert data["atomic"] is True
    assert path.read_text(encoding="utf-8") == "new"


def test_partial_read_cannot_replace_large_file_with_excerpt(tmp_path, monkeypatch):
    path = tmp_path / "large.py"
    original = "".join(f"line_{index} = {index}\n" for index in range(500))
    path.write_text(original, encoding="utf-8")
    context = _context(tmp_path, monkeypatch)
    context.context_ledger = ContextLedger()
    read_result = asyncio.run(
        read_file({"path": "large.py", "offset": 200, "limit": 40}, context)
    )
    version = _version_from_tool_message(read_result.content)

    result = asyncio.run(
        write_file(
            {
                "path": "large.py",
                "content": "line_200 = 200\nline_201 = 201\n",
                "expected_version": version,
            },
            context,
        )
    )

    assert result.is_error is True
    assert _json(result)["status"] == "FILE_READ_INCOMPLETE"
    assert path.read_text(encoding="utf-8") == original


def test_complete_read_allows_intentional_large_replacement(tmp_path, monkeypatch):
    path = tmp_path / "large.py"
    original = "".join(f"line_{index} = {index}\n" for index in range(500))
    path.write_text(original, encoding="utf-8")
    context = _context(tmp_path, monkeypatch)
    context.context_ledger = ContextLedger()
    read_result = asyncio.run(read_file({"path": "large.py", "limit": 1_000}, context))
    version = _version_from_tool_message(read_result.content)

    result = asyncio.run(
        write_file(
            {
                "path": "large.py",
                "content": "replacement = True\n",
                "expected_version": version,
            },
            context,
        )
    )

    assert result.is_error is False
    assert _json(result)["status"] == "WRITE_OK"
    assert path.read_text(encoding="utf-8") == "replacement = True\n"


def test_stale_version_rejects_overwrite_and_preserves_content(tmp_path, monkeypatch):
    path = tmp_path / "note.txt"
    path.write_text("external change", encoding="utf-8")

    result = asyncio.run(
        write_file(
            {
                "path": "note.txt",
                "content": "agent change",
                "expected_version": content_version(b"old"),
            },
            _context(tmp_path, monkeypatch),
        )
    )

    data = _json(result)
    assert result.is_error is True
    assert data["status"] == "FILE_VERSION_CONFLICT"
    assert data["retryable"] is True
    assert path.read_text(encoding="utf-8") == "external change"


def test_repeated_overwrite_of_same_content_is_a_noop(tmp_path, monkeypatch):
    path = tmp_path / "note.txt"
    path.write_text("old", encoding="utf-8")
    old_version = content_version(b"old")
    context = _context(tmp_path, monkeypatch)

    first = asyncio.run(
        write_file(
            {"path": "note.txt", "content": "new", "expected_version": old_version},
            context,
        )
    )
    second = asyncio.run(
        write_file(
            {"path": "note.txt", "content": "new", "expected_version": old_version},
            context,
        )
    )

    assert _json(first)["status"] == "WRITE_OK"
    assert _json(second)["status"] == "WRITE_NOOP"
    assert path.read_text(encoding="utf-8") == "new"


def test_missing_version_creates_once_but_never_overwrites(tmp_path, monkeypatch):
    context = _context(tmp_path, monkeypatch)

    created = asyncio.run(
        write_file(
            {"path": "new.txt", "content": "first", "expected_version": "missing"},
            context,
        )
    )
    conflict = asyncio.run(
        write_file(
            {"path": "new.txt", "content": "second", "expected_version": "missing"},
            context,
        )
    )

    assert _json(created)["status"] == "WRITE_OK"
    assert _json(conflict)["status"] == "FILE_VERSION_CONFLICT"
    assert (tmp_path / "new.txt").read_text(encoding="utf-8") == "first"


def test_warn_and_enforce_modes_handle_missing_expected_version(tmp_path, monkeypatch):
    path = tmp_path / "note.txt"
    path.write_text("old", encoding="utf-8")

    warning = asyncio.run(
        write_file(
            {"path": "note.txt", "content": "warn write"},
            _context(tmp_path, monkeypatch, mode="warn"),
        )
    )
    enforced = asyncio.run(
        write_file(
            {"path": "note.txt", "content": "blocked"},
            _context(tmp_path, monkeypatch, mode="enforce"),
        )
    )

    assert _json(warning)["warning"]
    assert _json(enforced)["status"] == "FILE_VERSION_REQUIRED"
    assert enforced.is_error is True
    assert path.read_text(encoding="utf-8") == "warn write"


def test_warn_mode_records_missing_version_in_audit_log(tmp_path, monkeypatch):
    path = tmp_path / "note.txt"
    path.write_text("old", encoding="utf-8")
    context = _context(tmp_path, monkeypatch, mode="warn")
    context.config.policy.hitl_mode = "never"
    registry = ToolRegistry()
    registry.register_all(get_builtin_tools())
    call = {
        "id": "write_1",
        "function": {
            "name": "write_file",
            "arguments": json.dumps({"path": "note.txt", "content": "new"}),
        },
    }

    result = asyncio.run(ToolExecutor(registry).execute_all([call], context))[0]
    audit = AuditLog(context.config.policy.audit_log_path).tail(1)[0]

    assert _json(result)["warning"]
    assert audit["outcome"] == "warning"
    assert "expected_version" in audit["details"]["warning"]


def test_repeated_append_with_stale_version_does_not_append_twice(tmp_path, monkeypatch):
    path = tmp_path / "note.txt"
    path.write_text("a", encoding="utf-8")
    version = content_version(b"a")
    context = _context(tmp_path, monkeypatch)
    payload = {
        "path": "note.txt",
        "content": "b",
        "append": True,
        "expected_version": version,
    }

    first = asyncio.run(write_file(payload, context))
    second = asyncio.run(write_file(payload, context))

    assert _json(first)["status"] == "WRITE_OK"
    assert _json(second)["status"] == "FILE_VERSION_CONFLICT"
    assert path.read_text(encoding="utf-8") == "ab"


def test_atomic_replace_failure_preserves_original_and_cleans_temp_file(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "note.txt"
    path.write_text("old", encoding="utf-8")
    version = content_version(b"old")

    def fail_replace(source, target):  # noqa: ARG001
        raise OSError("simulated replace failure")

    monkeypatch.setattr("paicli.tools.file_version.os.replace", fail_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        asyncio.run(
            write_file(
                {"path": "note.txt", "content": "new", "expected_version": version},
                _context(tmp_path, monkeypatch),
            )
        )

    assert path.read_text(encoding="utf-8") == "old"
    assert list(tmp_path.glob(".note.txt.*.tmp")) == []


class VersionConflictRecoveryClient:
    model_name = "fake-model"
    provider_name = "fake-provider"
    max_context_window = 1000

    def __init__(self, path):
        self.path = path
        self.calls = 0

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        self.calls += 1
        if self.calls == 1:
            yield _tool_call("read_1", "read_file", {"path": "note.txt"})
            yield {"type": "message_end", "stop_reason": "tool_use"}
            return
        if self.calls == 2:
            first_version = _version_from_tool_message(messages[-1].content)
            self.path.write_text("external", encoding="utf-8")
            yield _tool_call(
                "write_1",
                "write_file",
                {
                    "path": "note.txt",
                    "content": "agent",
                    "expected_version": first_version,
                },
            )
            yield {"type": "message_end", "stop_reason": "tool_use"}
            return
        if self.calls == 3:
            assert "FILE_VERSION_CONFLICT" in messages[-1].content
            yield _tool_call("read_2", "read_file", {"path": "note.txt"})
            yield {"type": "message_end", "stop_reason": "tool_use"}
            return
        if self.calls == 4:
            current_version = _version_from_tool_message(messages[-1].content)
            yield _tool_call(
                "write_2",
                "write_file",
                {
                    "path": "note.txt",
                    "content": "agent",
                    "expected_version": current_version,
                },
            )
            yield {"type": "message_end", "stop_reason": "tool_use"}
            return
        assert "WRITE_OK" in messages[-1].content
        yield {"type": "text_delta", "text": "Recovered from the file version conflict."}
        yield {"type": "message_end", "stop_reason": "end_turn"}


def _tool_call(call_id: str, name: str, arguments: dict) -> dict:
    return {
        "type": "tool_call_delta",
        "tool_call": {
            "index": 0,
            "id": call_id,
            "function": {"name": name, "arguments": json.dumps(arguments)},
        },
    }


def _version_from_tool_message(content: str) -> str:
    for line in content.splitlines():
        if line.startswith("version: "):
            return line.removeprefix("version: ")
    raise AssertionError("read_file result did not include a version")


def test_react_rereads_and_recovers_after_file_version_conflict(tmp_path, monkeypatch):
    path = tmp_path / "note.txt"
    path.write_text("original", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config = load_config(project_root=tmp_path)
    config.agent.stop_hook_enabled = False
    config.policy.hitl_mode = "never"
    registry = ToolRegistry()
    registry.register_all(get_builtin_tools())
    engine = QueryEngine(
        llm_client=VersionConflictRecoveryClient(path),
        tool_registry=registry,
        config=config,
        cwd=str(tmp_path),
    )

    result = asyncio.run(engine.ask_complete_async("replace note.txt safely"))

    assert result.completed is True
    assert result.text == "Recovered from the file version conflict."
    assert path.read_text(encoding="utf-8") == "agent"
