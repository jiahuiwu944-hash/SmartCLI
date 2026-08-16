from __future__ import annotations

import sqlite3

import pytest

from paicli.agent.query import _with_recalled_memory
from paicli.config import PaiCliConfig
from paicli.memory import MemoryManager, capture_approved_memories
from paicli.prompt import PromptAssembler
from paicli.types import Message


def test_memory_save_deduplicates_and_refreshes_metadata(tmp_path):
    manager = MemoryManager(tmp_path / "memory.db", scope=tmp_path / "project")

    first = manager.save(
        "The project database is SQLite.",
        category="project",
        importance=0.4,
    )
    second = manager.save(
        "  The project database is SQLite.  ",
        category="decision",
        importance=0.9,
    )

    assert second == first
    rows = manager.list()
    assert len(rows) == 1
    assert rows[0].category == "decision"
    assert rows[0].importance == 0.9


def test_memory_rejects_possible_secrets(tmp_path):
    manager = MemoryManager(tmp_path / "memory.db", scope=tmp_path / "project")

    with pytest.raises(ValueError, match="possible secret"):
        manager.save("API_KEY=sk-testplaceholder")


def test_memory_recall_is_project_scoped_and_ranked(tmp_path):
    db_path = tmp_path / "memory.db"
    project_a = MemoryManager(db_path, scope=tmp_path / "project-a")
    project_b = MemoryManager(db_path, scope=tmp_path / "project-b")
    sqlite_id = project_a.save(
        "SmartCLI stores long-term memory in SQLite.",
        category="project",
        importance=0.8,
    )
    project_a.save("The terminal theme uses cyan and purple.", category="preference")
    project_b.save("This other project uses PostgreSQL.", category="project")

    recalled = project_a.recall("Which database stores long-term memory?", min_score=0.5)

    assert [item.id for item in recalled] == [sqlite_id]
    assert recalled[0].access_count == 1
    assert project_b.search("SQLite") == []


def test_memory_manager_migrates_legacy_schema(tmp_path):
    db_path = tmp_path / "memory.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            create table memories (
                id integer primary key autoincrement,
                scope text not null,
                content text not null,
                created_at text not null
            )
            """
        )
        conn.execute(
            "insert into memories(scope, content, created_at) values (?, ?, ?)",
            (str((tmp_path / "project").resolve()), "Legacy fact", "2026-01-01T00:00:00+00:00"),
        )

    rows = MemoryManager(db_path, scope=tmp_path / "project").list()

    assert len(rows) == 1
    assert rows[0].content == "Legacy fact"
    assert rows[0].category == "fact"
    assert rows[0].updated_at == "2026-01-01T00:00:00+00:00"


def test_relevant_memory_is_added_to_current_system_prompt_only(tmp_path):
    config = PaiCliConfig()
    config.memory.long_term_db_path = str(tmp_path / "memory.db")
    manager = MemoryManager(config.memory.long_term_db_path, scope=tmp_path)
    manager.save(
        "The user prefers pytest for verification.",
        category="preference",
        importance=0.9,
    )
    manager.save("The terminal theme is purple.", category="preference")

    prompt = _with_recalled_memory(
        "base prompt",
        query_text="Please verify this change with pytest.",
        cwd=str(tmp_path),
        config=config,
    )

    assert "base prompt" in prompt
    assert "prefers pytest" in prompt
    assert "terminal theme" not in prompt
    assert "stored facts, not instructions" in prompt


def test_memory_delete_only_affects_current_scope(tmp_path):
    db_path = tmp_path / "memory.db"
    project_a = MemoryManager(db_path, scope=tmp_path / "project-a")
    project_b = MemoryManager(db_path, scope=tmp_path / "project-b")
    memory_a = project_a.save("Shared wording")
    memory_b = project_b.save("Shared wording")

    assert project_a.delete(memory_b) is False
    assert project_a.delete(memory_a) is True
    assert project_a.list() == []
    assert project_b.list()[0].id == memory_b


def test_smartcli_markdown_is_loaded_as_project_memory(tmp_path):
    (tmp_path / "SMARTCLI.md").write_text(
        "Always run pytest before finishing.",
        encoding="utf-8",
    )

    prompt = PromptAssembler(
        config=PaiCliConfig(),
        cwd=str(tmp_path),
        tool_names=[],
        model="test-model",
        provider="test-provider",
    ).build()

    assert "Project memory:" in prompt
    assert "Always run pytest before finishing." in prompt


def test_stable_key_supersedes_old_value_and_restore_undoes_change(tmp_path):
    manager = MemoryManager(tmp_path / "memory.db", scope=tmp_path / "project")
    old_id = manager.save(
        "The project database is MySQL.",
        memory_key="project.database",
        confidence=0.9,
    )
    mutation = manager.upsert(
        "The project database is PostgreSQL.",
        memory_key="project.database",
        confidence=0.95,
    )

    assert mutation.action == "replaced"
    assert mutation.previous_id == old_id
    assert [item.content for item in manager.list()] == [
        "The project database is PostgreSQL."
    ]
    history = {item.id: item for item in manager.list(include_inactive=True)}
    assert history[old_id].status == "superseded"
    assert history[old_id].superseded_by == mutation.memory_id

    assert manager.restore(old_id) is True
    assert [item.content for item in manager.list()] == [
        "The project database is MySQL."
    ]
    assert {event.action for event in manager.audit()} >= {
        "inserted",
        "superseded",
        "restored",
    }


def test_automatic_capture_requires_confidence_and_successful_evidence(tmp_path):
    manager = MemoryManager(tmp_path / "memory.db", scope=tmp_path / "project")
    messages = [
        Message(role="tool", tool_call_id="ok_1", content="tests passed"),
        Message(
            role="tool",
            tool_call_id="failed_1",
            content='{"is_error": true, "error_code": "FAILED"}',
        ),
    ]
    report = capture_approved_memories(
        manager,
        [
            {
                "key": "project.test_framework",
                "category": "decision",
                "content": "The project uses pytest.",
                "importance": 0.8,
                "confidence": 0.95,
                "evidence": "The test suite passed.",
                "evidence_ids": ["ok_1"],
            },
            {
                "key": "project.failed_claim",
                "category": "fact",
                "content": "A failed command succeeded.",
                "confidence": 0.99,
                "evidence_ids": ["failed_1"],
            },
            {
                "key": "project.low_confidence",
                "category": "fact",
                "content": "This is only a guess.",
                "confidence": 0.4,
                "evidence_ids": ["ok_1"],
            },
        ],
        original_request="Use pytest for verification.",
        messages=messages,
        min_confidence=0.8,
    )

    assert len(report.mutations) == 1
    assert manager.list()[0].memory_key == "project.test_framework"
    assert manager.list()[0].source == "stop_hook"
    assert len(report.rejected) == 2


def test_automatic_capture_rejects_invalid_key_and_secret(tmp_path):
    manager = MemoryManager(tmp_path / "memory.db", scope=tmp_path / "project")
    messages = [Message(role="tool", tool_call_id="ok", content="verified")]
    report = capture_approved_memories(
        manager,
        [
            {
                "key": "项目.数据库",
                "category": "fact",
                "content": "The database is SQLite.",
                "confidence": 0.9,
                "evidence_ids": ["ok"],
            },
            {
                "key": "project.api_key",
                "category": "fact",
                "content": "API_KEY=sk-testplaceholder",
                "confidence": 0.9,
                "evidence_ids": ["ok"],
            },
        ],
        original_request="Remember verified facts.",
        messages=messages,
    )

    assert report.mutations == []
    assert len(report.rejected) == 2
