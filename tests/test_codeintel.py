from __future__ import annotations

import asyncio
import json
from pathlib import Path

from paicli.codeintel import CodeNavigator, ContextLedger
from paicli.config import load_config
from paicli.tools import get_builtin_tools
from paicli.tools.base import ToolContext
from paicli.tools.builtins import read_file
from paicli.tools.executor import ToolExecutor
from paicli.tools.registry import ToolRegistry


def test_symbol_index_repo_map_references_and_incremental_update(tmp_path):
    source = tmp_path / "service.py"
    source.write_text(
        'class UserService:\n    """Users."""\n    def find_user(self):\n        return 1\n',
        encoding="utf-8",
    )
    caller = tmp_path / "caller.py"
    caller.write_text(
        "from service import UserService\nUserService().find_user()\n", encoding="utf-8"
    )

    navigator = CodeNavigator(tmp_path)
    first = navigator.update()
    symbols = navigator.find_symbol("UserService")
    references, truncated = navigator.find_references("find_user")

    assert first["changed"] == 2
    assert symbols[0].path == "service.py"
    assert any(item.path == "caller.py" for item in references)
    assert not truncated
    assert "UserService" in navigator.repo_map()

    second = navigator.update()
    assert second["changed"] == 0
    assert second["skipped"] == 2

    source.write_text("class RenamedService:\n    pass\n", encoding="utf-8")
    navigator.refresh_file("service.py")
    assert not navigator.find_symbol("UserService")
    assert navigator.find_symbol("RenamedService")


def test_search_code_uses_structured_results_and_gitignore(tmp_path):
    (tmp_path / ".gitignore").write_text("ignored.py\n", encoding="utf-8")
    (tmp_path / "main.py").write_text("def target_symbol():\n    return 1\n", encoding="utf-8")
    (tmp_path / "ignored.py").write_text("def target_symbol():\n    return 2\n", encoding="utf-8")

    results, _ = CodeNavigator(tmp_path).search("target_symbol")

    assert results
    assert all(item.path != "ignored.py" for item in results)
    assert any(item.reason in {"exact symbol match", "ripgrep text match"} for item in results)


def test_search_code_unifies_symbol_text_and_reference_modes(tmp_path):
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    (source_dir / "service.py").write_text(
        "class TargetService:\n    pass\n",
        encoding="utf-8",
    )
    (source_dir / "caller.py").write_text(
        "from service import TargetService\nTargetService()\n",
        encoding="utf-8",
    )
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    (other_dir / "duplicate.py").write_text(
        "class TargetService:\n    pass\n",
        encoding="utf-8",
    )
    navigator = CodeNavigator(tmp_path)

    symbols, _ = navigator.search("TargetService", mode="symbol", path="src")
    text, _ = navigator.search(r"TargetService\(\)", mode="text", path="src", regex=True)
    references, _ = navigator.search("TargetService", mode="references", path="src")

    assert symbols
    assert {item.path for item in symbols} == {str(Path("src/service.py"))}
    assert any(item.path == str(Path("src/caller.py")) for item in text)
    assert any(item.path == str(Path("src/caller.py")) for item in references)


def test_navigation_tool_surface_uses_unified_search_entry():
    names = {tool.name for tool in get_builtin_tools()}

    assert {"search_code", "repo_map", "document_symbols"} <= names
    assert "find_symbol" not in names
    assert "find_references" not in names
    assert "grep_code" not in names


def test_search_code_rejects_removed_fts_mode(tmp_path):
    navigator = CodeNavigator(tmp_path)

    try:
        navigator.search("approval", mode="semantic")
    except ValueError as exc:
        assert "unsupported search mode" in str(exc)
    else:
        raise AssertionError("removed semantic mode should be rejected")


def test_context_ledger_avoids_reinjecting_same_file_region(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "demo.py").write_text("one\ntwo\nthree\n", encoding="utf-8")
    context = ToolContext(
        cwd=str(tmp_path),
        config=load_config(project_root=tmp_path),
        context_ledger=ContextLedger(),
    )

    first = asyncio.run(read_file({"path": "demo.py", "offset": 1, "limit": 2}, context))
    second = asyncio.run(read_file({"path": "demo.py", "offset": 1, "limit": 2}, context))

    assert "1: one" in first.content
    assert "[CONTEXT_REUSE]" in second.content
    assert "1: one" not in second.content


def test_search_result_can_be_serialized_for_tool_context(tmp_path):
    (tmp_path / "demo.py").write_text("def calculate_total():\n    return 1\n", encoding="utf-8")
    results, truncated = CodeNavigator(tmp_path).search("calculate_total", limit=5)
    payload = {
        "matches": [
            {
                "path": item.path,
                "start_line": item.start_line,
                "reason": item.reason,
            }
            for item in results
        ],
        "truncated": truncated,
    }

    encoded = json.dumps(payload)
    assert json.loads(encoded)["matches"][0]["path"] == "demo.py"


def test_write_post_hook_refreshes_symbol_index(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config = load_config(project_root=tmp_path)
    config.policy.hitl_mode = "never"
    navigator = CodeNavigator(tmp_path)
    context = ToolContext(cwd=str(tmp_path), config=config, code_navigator=navigator)
    registry = ToolRegistry()
    registry.register_all(get_builtin_tools())
    call = {
        "id": "write_1",
        "type": "function",
        "function": {
            "name": "write_file",
            "arguments": json.dumps(
                {"path": "created.py", "content": "class CreatedByAgent:\n    pass\n"}
            ),
        },
    }

    result = asyncio.run(ToolExecutor(registry).execute_all([call], context))

    assert not result[0].is_error
    assert navigator.find_symbol("CreatedByAgent")
