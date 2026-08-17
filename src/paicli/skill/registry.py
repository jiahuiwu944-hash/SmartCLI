from __future__ import annotations

import json
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
SKILL_RESOURCE_DIRECTORIES = (
    "references",
    "scripts",
    "assets",
    "templates",
    "examples",
)
SUPPORTED_FRONTMATTER_KEYS = {
    "name",
    "description",
    "version",
    "author",
    "tags",
    "requires",
}


@dataclass(slots=True)
class SkillValidation:
    path: Path
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors


@dataclass(slots=True)
class Skill:
    name: str
    description: str
    path: Path
    content: str
    source: str = "project"
    version: str = ""
    tags: list[str] = field(default_factory=list)
    requires_tools: list[str] = field(default_factory=list)
    requires_mcp: list[str] = field(default_factory=list)
    enabled: bool = True

    @property
    def body(self) -> str:
        return _strip_frontmatter(self.content).strip()

    def resource_names(self, limit: int = 50) -> list[str]:
        """Return lazily readable files shipped beside SKILL.md."""
        root = self.path.parent.resolve()
        names: list[str] = []
        for directory in SKILL_RESOURCE_DIRECTORIES:
            candidate = root / directory
            if not candidate.is_dir():
                continue
            for path in sorted(candidate.rglob("*")):
                if not path.is_file() or path.is_symlink():
                    continue
                names.append(path.relative_to(root).as_posix())
                if len(names) >= limit:
                    return names
        return names


class SkillContextBuffer:
    def __init__(self, limit: int = 3):
        self.limit = limit
        self._items: OrderedDict[str, str] = OrderedDict()
        self._active: OrderedDict[str, None] = OrderedDict()

    def push(self, name: str | None, body: str | None) -> None:
        if not name or not body:
            return
        if name in self._items:
            del self._items[name]
        self._items[name] = body
        if name in self._active:
            del self._active[name]
        self._active[name] = None
        while len(self._items) > self.limit:
            evicted, _ = self._items.popitem(last=False)
            self._active.pop(evicted, None)

    def start_task(self) -> None:
        """Prevent instructions loaded by an earlier task leaking into a new task."""
        self.clear()

    def drain(self) -> str:
        if not self._items:
            return ""
        chunks = [
            f"## Loaded Skill: {name}\n{body.strip()}"
            for name, body in self._items.items()
            if body.strip()
        ]
        self._items.clear()
        return "\n\n".join(chunks)

    def drain_pending(self) -> str:
        """Consume skills that must be injected before the next model turn."""
        return self.drain()

    def clear(self) -> None:
        self._items.clear()
        self._active.clear()

    def is_empty(self) -> bool:
        return not self._items

    def size(self) -> int:
        return len(self._items)

    def is_active(self, name: str) -> bool:
        return name in self._active


class SkillStateStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or Path.home() / ".paicli" / "skills.json").expanduser()

    def disabled(self) -> set[str]:
        if not self.path.exists():
            return set()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return set()
        values = data.get("disabled") if isinstance(data, dict) else None
        if not isinstance(values, list):
            return set()
        return {str(item) for item in values if str(item).strip()}

    def disable(self, name: str) -> None:
        values = self.disabled()
        values.add(name)
        self._write(values)

    def enable(self, name: str) -> None:
        values = self.disabled()
        values.discard(name)
        self._write(values)

    def _write(self, disabled: set[str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"disabled": sorted(disabled)}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


class SkillRegistry:
    """Load SKILL.md files from built-in, user, and project locations."""

    def __init__(
        self,
        project_root: str | Path,
        *,
        builtin_root: str | Path | None = None,
        user_root: str | Path | None = None,
        state_store: SkillStateStore | None = None,
    ):
        self.project_root = Path(project_root).resolve()
        package_root = Path(__file__).resolve().parents[1]
        self.builtin_root = Path(builtin_root or package_root / "builtin_skills")
        self.user_root = Path(user_root or Path.home() / ".paicli" / "skills")
        self.project_skill_root = self.project_root / ".paicli" / "skills"
        self.state_store = state_store or SkillStateStore()
        self._skills: dict[str, Skill] | None = None

    def reload(self) -> None:
        self._skills = None

    def list(self) -> list[Skill]:
        return self.enabled_skills()

    def all_skills(self) -> list[Skill]:
        skills = self._load_all()
        return [skills[name] for name in sorted(skills)]

    def enabled_skills(self) -> list[Skill]:
        return [skill for skill in self.all_skills() if skill.enabled]

    def load(self, name: str, *, include_disabled: bool = False) -> Skill | None:
        skill = self._load_all().get(name)
        if not skill:
            return None
        if not include_disabled and not skill.enabled:
            return None
        return skill

    def enable(self, name: str) -> bool:
        if not self.load(name, include_disabled=True):
            return False
        self.state_store.enable(name)
        self.reload()
        return True

    def disable(self, name: str) -> bool:
        if not self.load(name, include_disabled=True):
            return False
        self.state_store.disable(name)
        self.reload()
        return True

    def index_text(self, max_chars: int = 4000, max_skills: int = 20) -> str:
        skills = self.enabled_skills()[:max_skills]
        if not skills:
            return ""
        lines = [
            "Available skills:",
            "Load a skill with load_skill(name) when its description matches the task.",
            "Use search_skills(query) when the visible list does not contain a match.",
        ]
        for skill in skills:
            description = " ".join(skill.description.split())
            if len(description) > 500:
                description = description[:497] + "..."
            lines.append(f"- {skill.name}: {description}")
        text = "\n".join(lines)
        return text[:max_chars]

    def search(self, query: str, limit: int = 10) -> list[Skill]:
        terms = [term for term in re.split(r"\W+", query.lower()) if term]
        if not terms:
            return []

        scored: list[tuple[int, str, Skill]] = []
        for skill in self.enabled_skills():
            name = skill.name.lower()
            description = skill.description.lower()
            tags = " ".join(skill.tags).lower()
            score = 0
            for term in terms:
                if term == name:
                    score += 100
                elif name.startswith(term):
                    score += 40
                elif term in name:
                    score += 25
                if term in tags:
                    score += 10
                if term in description:
                    score += 5
            if score:
                scored.append((score, skill.name, skill))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [item[2] for item in scored[: max(1, min(50, limit))]]

    def _load_all(self) -> dict[str, Skill]:
        if self._skills is not None:
            return self._skills
        disabled = self.state_store.disabled()
        skills: dict[str, Skill] = {}
        for source, root in [
            ("builtin", self.builtin_root),
            ("user", self.user_root),
            ("project", self.project_skill_root),
        ]:
            if not root.exists():
                continue
            for skill_file in sorted(root.glob("*/SKILL.md")):
                skill = self._load_skill_file(skill_file, source, disabled)
                if skill:
                    skills[skill.name] = skill
        self._skills = skills
        return skills

    def _load_skill_file(self, path: Path, source: str, disabled: set[str]) -> Skill | None:
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            return None
        validation = validate_skill_file(path, content=content)
        if not validation.valid:
            return None
        metadata = _parse_frontmatter(content)
        name = str(metadata["name"])
        description = str(metadata["description"])
        requires = metadata.get("requires")
        if not isinstance(requires, dict):
            requires = {}
        tags = _parse_string_list(metadata.get("tags"))
        return Skill(
            name=name,
            description=description,
            version=str(metadata.get("version") or ""),
            tags=tags,
            requires_tools=_parse_string_list(requires.get("tools")),
            requires_mcp=_parse_string_list(requires.get("mcp")),
            source=source,
            path=path,
            content=content,
            enabled=name not in disabled,
        )


def validate_skill_file(path: str | Path, *, content: str | None = None) -> SkillValidation:
    skill_path = Path(path)
    if skill_path.is_dir():
        skill_path = skill_path / "SKILL.md"
    result = SkillValidation(path=skill_path)
    if content is None:
        try:
            content = skill_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            result.errors.append(f"unable to read SKILL.md: {exc}")
            return result

    block = _frontmatter_block(content)
    if block is None:
        result.errors.append("SKILL.md must start with YAML frontmatter delimited by ---")
        return result
    try:
        metadata = yaml.safe_load(block)
    except yaml.YAMLError as exc:
        result.errors.append(f"invalid YAML frontmatter: {exc}")
        return result
    if not isinstance(metadata, dict):
        result.errors.append("frontmatter must be a YAML mapping")
        return result

    name = metadata.get("name")
    description = metadata.get("description")
    if not isinstance(name, str) or not name.strip():
        result.errors.append("frontmatter field 'name' must be a non-empty string")
    else:
        name = name.strip()
        if len(name) > 64 or not SKILL_NAME_PATTERN.fullmatch(name):
            result.errors.append("skill name must be <=64 lowercase letters, digits, or hyphens")
        if skill_path.parent.name != name:
            result.errors.append(
                f'skill directory "{skill_path.parent.name}" must match name "{name}"'
            )
    if not isinstance(description, str) or not description.strip():
        result.errors.append("frontmatter field 'description' must be a non-empty string")

    body = _strip_frontmatter(content).strip()
    if not body:
        result.errors.append("SKILL.md body must not be empty")
    if len(body.splitlines()) > 500:
        result.warnings.append(
            "SKILL.md body exceeds 500 lines; move detailed material into references/"
        )
    unknown = sorted(str(key) for key in metadata if key not in SUPPORTED_FRONTMATTER_KEYS)
    if unknown:
        result.warnings.append(f"unknown frontmatter fields: {', '.join(unknown)}")

    requires = metadata.get("requires")
    if requires is not None and not isinstance(requires, dict):
        result.errors.append("frontmatter field 'requires' must be a mapping")
    elif isinstance(requires, dict):
        for key in ("tools", "mcp"):
            value = requires.get(key)
            if value is not None and not _is_string_list(value):
                result.errors.append(f"requires.{key} must be a string or list of strings")
    if metadata.get("tags") is not None and not _is_string_list(metadata["tags"]):
        result.errors.append("frontmatter field 'tags' must be a string or list of strings")
    return result


def _frontmatter_block(content: str) -> str | None:
    if not content.startswith("---"):
        return None
    match = re.match(r"^---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|$)", content, re.S)
    return match.group(1) if match else None


def _parse_frontmatter(content: str) -> dict[str, Any]:
    block = _frontmatter_block(content)
    if block is None:
        return {}
    try:
        metadata = yaml.safe_load(block)
    except yaml.YAMLError:
        return {}
    return metadata if isinstance(metadata, dict) else {}


def _strip_frontmatter(content: str) -> str:
    block = _frontmatter_block(content)
    if block is None:
        return content
    return re.sub(
        r"^---[ \t]*\r?\n.*?\r?\n---[ \t]*(?:\r?\n|$)",
        "",
        content,
        count=1,
        flags=re.S,
    )


def _is_string_list(value: Any) -> bool:
    return isinstance(value, str) or (
        isinstance(value, list) and all(isinstance(item, str) for item in value)
    )


def _parse_string_list(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [item.strip() for item in raw if isinstance(item, str) and item.strip()]
    if not isinstance(raw, str):
        return []
    value = raw.strip()
    if not value:
        return []
    return [item.strip().strip('"').strip("'") for item in value.split(",") if item.strip()]
