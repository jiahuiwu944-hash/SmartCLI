from __future__ import annotations

import asyncio
from pathlib import Path

from paicli.config import load_config
from paicli.skill import SkillContextBuffer, SkillRegistry, SkillStateStore
from paicli.tools import ToolRegistry, get_builtin_tools
from paicli.tools.base import ToolContext
from paicli.tools.builtins import load_skill, read_skill_resource


def test_skill_registry_layers_and_state(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    builtin = tmp_path / "builtin"
    user = tmp_path / "user"
    project = tmp_path / "project"
    _write_skill(builtin, "web-access", "builtin desc", "v0")
    _write_skill(user, "web-access", "user desc", "v1")
    _write_skill(project / ".paicli" / "skills", "project-only", "project desc", "v2")
    state = SkillStateStore(tmp_path / "skills.json")
    state.disable("web-access")

    registry = SkillRegistry(
        project,
        builtin_root=builtin,
        user_root=user,
        state_store=state,
    )

    assert [skill.name for skill in registry.all_skills()] == ["project-only", "web-access"]
    assert registry.load("web-access") is None
    assert registry.load("web-access", include_disabled=True).source == "user"
    assert [skill.name for skill in registry.enabled_skills()] == ["project-only"]

    assert registry.enable("web-access")
    assert registry.load("web-access").description == "user desc"


def test_skill_context_buffer_is_one_shot_and_capped():
    buffer = SkillContextBuffer(limit=3)
    buffer.push("a", "A")
    buffer.push("b", "B")
    buffer.push("c", "C")
    buffer.push("d", "D")

    drained = buffer.drain()

    assert "Loaded Skill: a" not in drained
    assert "Loaded Skill: b" in drained
    assert "Loaded Skill: c" in drained
    assert "Loaded Skill: d" in drained
    assert buffer.drain() == ""


def test_load_skill_pushes_body_into_context_buffer(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    _write_skill(tmp_path / ".paicli" / "skills", "demo", "demo desc", "v1", body="demo body")
    config = load_config(project_root=tmp_path)
    buffer = SkillContextBuffer()
    context = ToolContext(cwd=str(tmp_path), config=config, skill_context_buffer=buffer)

    result = asyncio.run(load_skill({"name": "demo"}, context))

    assert not result.is_error
    drained = buffer.drain()
    assert "Loaded Skill: demo" in drained
    assert "demo body" in drained


def test_skill_resources_are_advertised_and_read_lazily(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    _write_skill(tmp_path / ".paicli" / "skills", "demo", "demo desc", "v1")
    resource = tmp_path / ".paicli" / "skills" / "demo" / "references" / "guide.md"
    resource.parent.mkdir(parents=True)
    resource.write_text("detailed guide", encoding="utf-8")
    config = load_config(project_root=tmp_path)
    buffer = SkillContextBuffer()
    registry = ToolRegistry()
    registry.register_all(get_builtin_tools())
    context = ToolContext(
        cwd=str(tmp_path),
        config=config,
        skill_context_buffer=buffer,
        tool_registry=registry,
    )

    loaded = asyncio.run(load_skill({"name": "demo"}, context))
    instructions = buffer.drain_pending()
    result = asyncio.run(
        read_skill_resource({"name": "demo", "path": "references/guide.md"}, context)
    )

    assert not loaded.is_error
    assert "references/guide.md" in instructions
    assert result.content == "detailed guide"


def test_skill_dependency_check_reports_missing_tool_and_mcp(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    skill_dir = tmp_path / ".paicli" / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    skill_dir.joinpath("SKILL.md").write_text(
        "---\nname: demo\ndescription: demo\nrequires:\n"
        "  tools: [read_file, unavailable_tool]\n"
        "  mcp: [chrome-devtools]\n---\nbody\n",
        encoding="utf-8",
    )
    config = load_config(project_root=tmp_path)
    registry = ToolRegistry()
    registry.register_all(get_builtin_tools())
    context = ToolContext(cwd=str(tmp_path), config=config, tool_registry=registry)

    result = asyncio.run(load_skill({"name": "demo"}, context))

    assert result.is_error
    assert "tool:unavailable_tool" in result.content
    assert "mcp:chrome-devtools" in result.content


def test_skill_buffer_start_task_discards_stale_instructions():
    buffer = SkillContextBuffer()
    buffer.push("old", "stale")

    buffer.start_task()

    assert buffer.is_empty()


def _write_skill(
    root: Path,
    name: str,
    desc: str,
    version: str,
    *,
    body: str | None = None,
) -> None:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_dir.joinpath("SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {desc}\nversion: {version}\n---\n"
        f"{body or f'body for {name}'}\n",
        encoding="utf-8",
    )
