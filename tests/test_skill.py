from __future__ import annotations

import asyncio
from pathlib import Path

from paicli.config import load_config
from paicli.skill import (
    SkillContextBuffer,
    SkillRegistry,
    SkillStateStore,
    validate_skill_file,
)
from paicli.tools import ToolRegistry, get_builtin_tools
from paicli.tools.base import ToolContext
from paicli.tools.builtins import (
    copy_skill_resource,
    load_skill,
    read_skill_resource,
    search_skills,
)


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
    assert not buffer.is_active("a")
    assert buffer.is_active("d")


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


def test_load_skill_injects_complete_long_body(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    body = "start\n" + ("x" * 6_000) + "\nimportant-final-instruction"
    _write_skill(tmp_path / ".paicli" / "skills", "demo", "demo desc", "v1", body=body)
    config = load_config(project_root=tmp_path)
    buffer = SkillContextBuffer()
    context = ToolContext(cwd=str(tmp_path), config=config, skill_context_buffer=buffer)

    result = asyncio.run(load_skill({"name": "demo"}, context))
    instructions = buffer.drain_pending()

    assert not result.is_error
    assert "important-final-instruction" in instructions
    assert "truncated" not in instructions


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


def test_skill_resource_requires_activation_and_supports_paging(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    _write_skill(tmp_path / ".paicli" / "skills", "demo", "demo desc", "v1")
    skill_dir = tmp_path / ".paicli" / "skills" / "demo"
    resource = skill_dir / "scripts" / "probe.py"
    resource.parent.mkdir(parents=True)
    resource.write_text("a" * 700 + "END", encoding="utf-8")
    asset = skill_dir / "assets" / "template.txt"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"\x00\x01template")
    config = load_config(project_root=tmp_path)
    buffer = SkillContextBuffer()
    context = ToolContext(cwd=str(tmp_path), config=config, skill_context_buffer=buffer)

    denied = asyncio.run(
        read_skill_resource({"name": "demo", "path": "scripts/probe.py"}, context)
    )
    loaded = asyncio.run(load_skill({"name": "demo"}, context))
    advertised = buffer.drain_pending()
    first = asyncio.run(
        read_skill_resource(
            {"name": "demo", "path": "scripts/probe.py", "max_chars": 500}, context
        )
    )
    second = asyncio.run(
        read_skill_resource(
            {
                "name": "demo",
                "path": "scripts/probe.py",
                "max_chars": 500,
                "offset": 500,
            },
            context,
        )
    )
    copied = asyncio.run(
        copy_skill_resource(
            {
                "name": "demo",
                "path": "assets/template.txt",
                "destination": "generated/template.bin",
            },
            context,
        )
    )

    assert denied.is_error
    assert not loaded.is_error
    assert "scripts/probe.py" in advertised
    assert "assets/template.txt" in advertised
    assert "offset=500" in first.content
    assert second.content.endswith("END")
    assert not copied.is_error
    assert (tmp_path / "generated" / "template.bin").read_bytes() == b"\x00\x01template"


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


def test_skill_feature_flag_hides_and_rejects_skill_tools(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    _write_skill(tmp_path / ".paicli" / "skills", "demo", "demo desc", "v1")
    config = load_config(project_root=tmp_path)
    config.features.skill = False
    context = ToolContext(
        cwd=str(tmp_path), config=config, skill_context_buffer=SkillContextBuffer()
    )

    names = {tool.name for tool in get_builtin_tools(skill_enabled=False)}
    result = asyncio.run(load_skill({"name": "demo"}, context))

    assert {
        "load_skill",
        "search_skills",
        "read_skill_resource",
        "copy_skill_resource",
    }.isdisjoint(names)
    assert result.is_error
    assert "disabled" in result.content.lower()


def test_skill_search_finds_items_beyond_prompt_index_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    root = tmp_path / ".paicli" / "skills"
    for index in range(25):
        _write_skill(root, f"skill-{index}", f"ordinary workflow {index}", "v1")
    _write_skill(root, "zebra-research", "special striped-animal workflow", "v1")
    config = load_config(project_root=tmp_path)
    context = ToolContext(cwd=str(tmp_path), config=config)

    assert "zebra-research" not in SkillRegistry(tmp_path).index_text(max_skills=20)
    result = asyncio.run(search_skills({"query": "striped-animal"}, context))

    assert not result.is_error
    assert "zebra-research" in result.content


def test_skill_validation_rejects_invalid_metadata_and_accepts_builtin(tmp_path):
    invalid = tmp_path / "Wrong_Name" / "SKILL.md"
    invalid.parent.mkdir(parents=True)
    invalid.write_text("---\nname: Wrong_Name\ndescription: ''\n---\n", encoding="utf-8")

    result = validate_skill_file(invalid)
    builtin = validate_skill_file(
        Path(__file__).parents[1] / "src" / "paicli" / "builtin_skills" / "web-access"
    )

    assert not result.valid
    assert any("skill name" in error for error in result.errors)
    assert any("description" in error for error in result.errors)
    assert builtin.valid


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
