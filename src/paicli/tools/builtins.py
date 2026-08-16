from __future__ import annotations

import asyncio
import glob as glob_module
import json
import locale
import os
import re
from pathlib import Path
from typing import Any

from paicli.codeintel import CodeNavigator
from paicli.lsp import diagnose_file as run_diagnostics
from paicli.memory import MemoryManager
from paicli.policy import CommandGuard, PathGuard
from paicli.skill import SkillRegistry
from paicli.snapshot import SnapshotService
from paicli.tools.base import Tool, ToolContext, ToolResult, object_schema
from paicli.tools.file_version import (
    MISSING_VERSION,
    FileChangedDuringWriteError,
    atomic_write,
    content_version,
    file_version,
)
from paicli.web import fetch_url, search_web


def get_builtin_tools() -> list[Tool]:
    tools = [
        Tool(
            name="read_file",
            description=(
                "Read a text file and return its current version. Pass that version to "
                "write_file when changing an existing file."
            ),
            parameters=object_schema(
                {
                    "path": {"type": "string", "description": "Path to read"},
                    "offset": {"type": "number", "description": "Start line, 1-based"},
                    "limit": {"type": "number", "description": "Maximum number of lines"},
                },
                ["path"],
            ),
            required_keys=["path"],
            handler=read_file,
        ),
        Tool(
            name="write_file",
            description=(
                "Atomically write a UTF-8 file. Before overwriting or appending, read the "
                "file and pass its version as expected_version."
            ),
            parameters=object_schema(
                {
                    "path": {"type": "string", "description": "Path to write"},
                    "content": {"type": "string", "description": "File content"},
                    "append": {"type": "boolean", "description": "Append instead of overwrite"},
                    "expected_version": {
                        "type": "string",
                        "description": (
                            "Version returned by read_file, or 'missing' when the file must "
                            "not already exist"
                        ),
                    },
                },
                ["path", "content"],
            ),
            required_keys=["path", "content"],
            handler=write_file,
            is_read_only=False,
            is_concurrency_safe=False,
            danger_level="medium",
        ),
        Tool(
            name="list_dir",
            description="List entries in a directory inside the current workspace.",
            parameters=object_schema(
                {"path": {"type": "string", "description": "Directory path"}},
                ["path"],
            ),
            required_keys=["path"],
            handler=list_dir,
        ),
        Tool(
            name="glob",
            description="Find files by glob pattern inside the current workspace.",
            parameters=object_schema(
                {
                    "pattern": {"type": "string", "description": "Glob pattern"},
                    "limit": {"type": "number", "description": "Maximum results"},
                },
                ["pattern"],
            ),
            required_keys=["pattern"],
            handler=glob_files,
        ),
        Tool(
            name="glob_files",
            description="Alias of glob. Find files by glob pattern inside the current workspace.",
            parameters=object_schema(
                {
                    "pattern": {"type": "string", "description": "Glob pattern"},
                    "limit": {"type": "number", "description": "Maximum results"},
                },
                ["pattern"],
            ),
            required_keys=["pattern"],
            handler=glob_files,
        ),
        Tool(
            name="bash",
            description=_shell_tool_description(),
            parameters=object_schema(
                {
                    "command": {"type": "string", "description": "Shell command"},
                    "timeout": {"type": "number", "description": "Timeout seconds"},
                },
                ["command"],
            ),
            required_keys=["command"],
            handler=bash,
            is_read_only=False,
            is_concurrency_safe=False,
            danger_level="high",
            requires_approval=True,
        ),
        Tool(
            name="execute_command",
            description=f"Alias of bash. {_shell_tool_description()}",
            parameters=object_schema(
                {
                    "command": {"type": "string", "description": "Shell command"},
                    "timeout": {"type": "number", "description": "Timeout seconds"},
                },
                ["command"],
            ),
            required_keys=["command"],
            handler=bash,
            is_read_only=False,
            is_concurrency_safe=False,
            danger_level="high",
            requires_approval=True,
        ),
        Tool(
            name="web_search",
            description=(
                "Search the web for current information. Returns titles, URLs, and snippets."
            ),
            parameters=object_schema(
                {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "number", "description": "Maximum result count"},
                },
                ["query"],
            ),
            required_keys=["query"],
            handler=web_search,
        ),
        Tool(
            name="web_fetch",
            description="Fetch a public HTTP/HTTPS page and return readable text.",
            parameters=object_schema(
                {
                    "url": {"type": "string", "description": "URL to fetch"},
                    "max_length": {"type": "number", "description": "Maximum returned characters"},
                },
                ["url"],
            ),
            required_keys=["url"],
            handler=web_fetch,
        ),
        Tool(
            name="save_memory",
            description=(
                "Save a verified, stable cross-session fact to project memory. Do not save "
                "secrets, guesses, raw tool output, or temporary task progress."
            ),
            parameters=object_schema(
                {
                    "content": {"type": "string", "description": "Concise fact to remember"},
                    "category": {
                        "type": "string",
                        "enum": [
                            "fact",
                            "preference",
                            "project",
                            "decision",
                            "constraint",
                            "solution",
                        ],
                        "description": "Memory category; defaults to fact",
                    },
                      "importance": {
                          "type": "number",
                          "description": "Long-term importance from 0.0 to 1.0",
                      },
                      "key": {
                          "type": "string",
                          "description": (
                              "Optional stable dotted key used to replace an older fact"
                          ),
                      },
                      "confidence": {
                          "type": "number",
                          "description": "Evidence-backed confidence from 0.0 to 1.0",
                      },
                      "evidence": {
                          "type": "string",
                          "description": "Concise source or verification evidence",
                      },
                },
                ["content"],
            ),
            required_keys=["content"],
            handler=save_memory,
            is_read_only=False,
            is_concurrency_safe=False,
            danger_level="medium",
        ),
        Tool(
            name="search_memory",
            description=(
                "Search relevant long-term memories for this project. Returned facts may be "
                "stale and must be verified before changing code."
            ),
            parameters=object_schema(
                {
                    "query": {"type": "string", "description": "Memory search query"},
                    "limit": {"type": "number", "description": "Maximum results"},
                },
                ["query"],
            ),
            required_keys=["query"],
            handler=search_memory,
        ),
        Tool(
            name="load_skill",
            description=(
                "Activate a named SmartCLI skill for the current task. The instructions are "
                "injected before the next ReAct turn."
            ),
            parameters=object_schema(
                {"name": {"type": "string", "description": "Skill name"}},
                ["name"],
            ),
            required_keys=["name"],
            handler=load_skill,
        ),
        Tool(
            name="read_skill_resource",
            description=(
                "Read one reference, template, or example advertised by an activated skill. "
                "Use this only when the core SKILL.md instructions require that resource."
            ),
            parameters=object_schema(
                {
                    "name": {"type": "string", "description": "Skill name"},
                    "path": {
                        "type": "string",
                        "description": "Relative resource path shown by load_skill",
                    },
                    "max_chars": {
                        "type": "number",
                        "description": "Maximum characters to return; defaults to 8000",
                    },
                },
                ["name", "path"],
            ),
            required_keys=["name", "path"],
            handler=read_skill_resource,
        ),
        Tool(
            name="search_code",
            description=(
                "Unified code navigation for exact definitions, ripgrep text, and symbol "
                "references. Use repo_map first only when the project structure is unknown."
            ),
            parameters=object_schema(
                {
                    "query": {"type": "string", "description": "Search query"},
                    "mode": {
                        "type": "string",
                        "enum": ["auto", "symbol", "text", "references"],
                        "description": "Navigation strategy; defaults to auto",
                    },
                    "path": {"type": "string", "description": "Workspace path to search"},
                    "kind": {
                        "type": "string",
                        "description": "Optional class/function/method filter for symbol mode",
                    },
                    "regex": {
                        "type": "boolean",
                        "description": "Treat query as regex in text mode",
                    },
                    "limit": {"type": "number", "description": "Maximum matches"},
                },
                ["query"],
            ),
            required_keys=["query"],
            handler=search_code,
        ),
        Tool(
            name="repo_map",
            description="Return a compact repository map of files and important symbols.",
            parameters=object_schema(
                {"max_chars": {"type": "number", "description": "Maximum output characters"}}
            ),
            handler=repo_map,
        ),
        Tool(
            name="document_symbols",
            description="List indexed classes, methods, and functions in one source file.",
            parameters=object_schema(
                {
                    "path": {"type": "string", "description": "Source file"},
                    "limit": {"type": "number", "description": "Maximum symbols"},
                },
                ["path"],
            ),
            required_keys=["path"],
            handler=document_symbols,
        ),
        Tool(
            name="diagnose_file",
            description="Run available local syntax diagnostics for a source file.",
            parameters=object_schema(
                {"path": {"type": "string", "description": "Source file"}}, ["path"]
            ),
            required_keys=["path"],
            handler=diagnose_source_file,
        ),
        Tool(
            name="revert_turn",
            description="Restore the workspace to a previous SmartCLI side-history snapshot.",
            parameters=object_schema(
                {"snapshot": {"type": "string", "description": "Snapshot id or 1-based index"}},
                ["snapshot"],
            ),
            required_keys=["snapshot"],
            handler=revert_turn,
            is_read_only=False,
            is_concurrency_safe=False,
            danger_level="high",
            requires_approval=True,
        ),
    ]
    return tools


async def read_file(payload: dict[str, Any], context: ToolContext) -> ToolResult:
    path = _resolve_path(context, str(payload["path"]))
    offset = max(int(payload.get("offset") or 1), 1)
    limit = int(payload.get("limit") or 500)
    raw_content = path.read_bytes()
    version = content_version(raw_content)
    content = raw_content.decode("utf-8", errors="replace").splitlines()
    selected = content[offset - 1 : offset - 1 + limit]
    numbered = "\n".join(f"{idx + offset}: {line}" for idx, line in enumerate(selected))
    relative_path = path.relative_to(context.cwd)
    metadata = (
        "[FILE_METADATA]\n"
        f"path: {relative_path}\n"
        f"version: {version}\n"
        f"size: {len(raw_content)}\n"
        "[FILE_CONTENT]"
    )
    end_line = offset + len(selected) - 1
    ledger = context.context_ledger
    if ledger is not None and ledger.seen(str(relative_path), offset, end_line, version):
        return ToolResult(
            f"{metadata}\n[CONTEXT_REUSE] This exact file version and line range was already "
            "read in the current run. Reuse the existing context instead of consuming it again.",
            display_summary=f"Reuse {relative_path}:{offset}-{end_line}",
        )
    if ledger is not None:
        ledger.record(str(relative_path), offset, end_line, version)
    return ToolResult(
        f"{metadata}\n{numbered}",
        display_summary=f"Read {relative_path} ({version[:19]}...)",
    )


async def write_file(payload: dict[str, Any], context: ToolContext) -> ToolResult:
    path = _resolve_path(context, str(payload["path"]))
    content = str(payload["content"])
    content_bytes = content.encode("utf-8")
    append = bool(payload.get("append"))
    expected_version = payload.get("expected_version")
    expected_version = str(expected_version) if expected_version is not None else None
    current_content = path.read_bytes() if path.exists() else b""
    current_version = file_version(path)
    desired_content = current_content + content_bytes if append else content_bytes
    desired_version = content_version(desired_content)
    if len(desired_content) > 5 * 1024 * 1024:
        return ToolResult("write_file rejected: content exceeds 5MB", is_error=True)

    relative_path = path.relative_to(context.cwd)
    if current_version == desired_version:
        return ToolResult(
            _file_result(
                "WRITE_NOOP",
                relative_path,
                version=current_version,
                message="Target content is already present; no file write was performed.",
            ),
            display_summary=f"No change {relative_path}",
        )

    check_mode = _file_version_check_mode(context)
    if check_mode != "off" and expected_version is not None:
        if expected_version != current_version:
            return _version_conflict_result(
                relative_path,
                expected_version=expected_version,
                actual_version=current_version,
            )
    elif check_mode == "enforce" and current_version != MISSING_VERSION:
        return ToolResult(
            _file_result(
                "FILE_VERSION_REQUIRED",
                relative_path,
                actual_version=current_version,
                retryable=True,
                suggested_action=(
                    "Call read_file and pass its version as expected_version before overwriting "
                    "or appending."
                ),
            ),
            is_error=True,
            display_summary=f"Version required {relative_path}",
        )

    incomplete_read = _incomplete_read_for_destructive_overwrite(
        context=context,
        relative_path=relative_path,
        current_content=current_content,
        desired_content=desired_content,
        current_version=current_version,
        append=append,
    )
    if incomplete_read:
        return ToolResult(
            _file_result(
                "FILE_READ_INCOMPLETE",
                relative_path,
                actual_version=current_version,
                current_size=len(current_content),
                requested_size=len(desired_content),
                retryable=True,
                suggested_action=(
                    "This overwrite would remove a large portion of an existing file, but the "
                    "current file version was only read partially. Read the complete file with "
                    "read_file (one or more contiguous ranges), regenerate the full content, and "
                    "retry with the same expected_version."
                ),
            ),
            is_error=True,
            display_summary=f"Complete read required {relative_path}",
        )

    warning = None
    if check_mode == "warn" and expected_version is None and current_version != MISSING_VERSION:
        warning = "Existing file was written without expected_version."

    try:
        if context.config.tools.atomic_file_write:
            atomic_write(path, desired_content, observed_version=current_version)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(desired_content)
    except FileChangedDuringWriteError as exc:
        return _version_conflict_result(
            relative_path,
            expected_version=current_version,
            actual_version=exc.actual_version,
        )

    new_version = file_version(path)
    rel = path.relative_to(context.cwd)
    diagnostics = run_diagnostics(path)
    result = _file_result(
        "WRITE_OK",
        relative_path,
        previous_version=current_version,
        version=new_version,
        atomic=context.config.tools.atomic_file_write,
        warning=warning,
        diagnostics=diagnostics or None,
    )
    return ToolResult(result, display_summary=f"Wrote {rel} ({new_version[:19]}...)")


def _file_version_check_mode(context: ToolContext) -> str:
    mode = str(context.config.tools.file_version_check or "warn").lower()
    return mode if mode in {"off", "warn", "enforce"} else "warn"


def _incomplete_read_for_destructive_overwrite(
    *,
    context: ToolContext,
    relative_path: Path,
    current_content: bytes,
    desired_content: bytes,
    current_version: str,
    append: bool,
) -> bool:
    """Guard against replacing a whole file with a partially read excerpt."""

    ledger = context.context_ledger
    if append or ledger is None or len(current_content) < 4 * 1024:
        return False
    if len(desired_content) * 100 >= len(current_content) * 80:
        return False
    line_count = len(current_content.decode("utf-8", errors="replace").splitlines())
    return not ledger.covers(
        str(relative_path),
        1,
        max(1, line_count),
        current_version,
    )


def _version_conflict_result(
    path: Path,
    *,
    expected_version: str,
    actual_version: str,
) -> ToolResult:
    return ToolResult(
        _file_result(
            "FILE_VERSION_CONFLICT",
            path,
            expected_version=expected_version,
            actual_version=actual_version,
            retryable=True,
            suggested_action=(
                "Call read_file again, regenerate the change from the latest content, and retry "
                "with the new version. Do not force an overwrite."
            ),
        ),
        is_error=True,
        display_summary=f"Version conflict {path}",
    )


def _file_result(status: str, path: Path, **details: Any) -> str:
    return json.dumps(
        {"status": status, "path": str(path), **details},
        ensure_ascii=False,
        indent=2,
    )


async def list_dir(payload: dict[str, Any], context: ToolContext) -> ToolResult:
    path = _resolve_path(context, str(payload["path"]))
    if not path.is_dir():
        return ToolResult(f"Not a directory: {path}", is_error=True)
    rows = []
    for child in sorted(path.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
        marker = "/" if child.is_dir() else ""
        rows.append(f"{child.name}{marker}")
    return ToolResult("\n".join(rows) or "(empty directory)")


async def glob_files(payload: dict[str, Any], context: ToolContext) -> ToolResult:
    root = Path(context.cwd).resolve()
    pattern = str(payload["pattern"])
    if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
        return ToolResult("glob pattern must stay inside workspace", is_error=True)
    limit = int(payload.get("limit") or 100)
    matches = glob_module.glob(str(root / pattern), recursive=True)
    rels = []
    for match in sorted(matches):
        path = Path(match).resolve()
        try:
            rels.append(str(path.relative_to(root)))
        except ValueError:
            continue
        if len(rels) >= limit:
            break
    return ToolResult("\n".join(rels) or "(no matches)")


async def grep(payload: dict[str, Any], context: ToolContext) -> ToolResult:
    pattern = str(payload["pattern"])
    limit = int(payload.get("limit") or 100)
    use_regex = bool(payload.get("regex", True))
    try:
        matches, truncated = _navigator(context).scanner.search(
            pattern,
            path=str(payload.get("path") or "."),
            regex=use_regex,
            limit=limit,
        )
    except (RuntimeError, re.error) as exc:
        return ToolResult(f"grep failed: {exc}", is_error=True)
    return ToolResult(_search_result_json(matches, truncated=truncated, query=pattern))


async def bash(payload: dict[str, Any], context: ToolContext) -> ToolResult:
    command = str(payload["command"])
    CommandGuard(context.config.policy.command_blacklist).validate(command)
    timeout = float(payload.get("timeout") or context.config.tools.timeout)
    proc = await asyncio.create_subprocess_shell(
        command,
        cwd=context.cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=os.environ.copy(),
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return ToolResult(f"Command timed out after {timeout:.0f}s", is_error=True)
    output = _join_shell_output(stdout, stderr)
    if len(output) > 20_000:
        output = output[:20_000] + "\n... [truncated]"
    return ToolResult(
        output or f"(exit {proc.returncode}, no output)",
        is_error=proc.returncode != 0,
    )


def _shell_tool_description() -> str:
    shell = "Windows Command Prompt (cmd.exe)" if os.name == "nt" else "/bin/sh"
    return (
        f"Execute a command with {shell} in the current workspace. "
        "Use it for tests, diagnostics, builds, and read-only inspection; do not use shell "
        "redirection or patch scripts to modify workspace source files. "
        "A non-zero exit code is reported as an error; when a probe may legitimately find "
        "nothing, print a clear no-match result and exit successfully."
    )


def _join_shell_output(stdout: bytes, stderr: bytes) -> str:
    parts = [_decode_shell_output(value).strip() for value in (stdout, stderr) if value]
    return "\n".join(part for part in parts if part)


def _decode_shell_output(value: bytes) -> str:
    if not value:
        return ""

    encodings = ["utf-8", locale.getpreferredencoding(False)]
    if os.name == "nt":
        encodings.extend(["mbcs", "gb18030"])

    tried: set[str] = set()
    for encoding in encodings:
        normalized = encoding.lower()
        if normalized in tried:
            continue
        tried.add(normalized)
        try:
            return value.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            continue
    return value.decode("utf-8", errors="replace")


async def web_search(payload: dict[str, Any], _context: ToolContext) -> ToolResult:
    max_results = int(payload.get("max_results") or payload.get("maxResults") or 5)
    try:
        results = await search_web(str(payload["query"]), max_results=max_results)
    except Exception as exc:  # noqa: BLE001
        return ToolResult(f"Search error: {exc}", is_error=True)
    if not results:
        return ToolResult(f'No search results found for "{payload["query"]}".')
    content = "\n\n".join(
        f"{index}. {result.title}\n{result.url}\n{result.snippet}"
        for index, result in enumerate(results, start=1)
    )
    return ToolResult(content, display_summary=f"Search: {len(results)} results")


async def web_fetch(payload: dict[str, Any], _context: ToolContext) -> ToolResult:
    max_length = int(payload.get("max_length") or payload.get("maxLength") or 10_000)
    try:
        content = await fetch_url(str(payload["url"]), max_length=max_length)
    except Exception as exc:  # noqa: BLE001
        return ToolResult(f"Fetch error: {exc}", is_error=True)
    return ToolResult(content, display_summary=f"Fetched {payload['url']}")


async def save_memory(payload: dict[str, Any], context: ToolContext) -> ToolResult:
    if not context.config.features.memory or not context.config.memory.long_term_enabled:
        return ToolResult("Long-term memory is disabled.", is_error=True)
    manager = MemoryManager(context.config.memory.long_term_db_path, scope=context.cwd)
    try:
        memory_id = manager.save(
            str(payload["content"]),
            category=str(payload.get("category") or "fact"),
            importance=payload.get("importance", 0.5),
            source="agent",
            memory_key=str(payload.get("key") or ""),
            confidence=payload.get("confidence", 0.5),
            evidence=str(payload.get("evidence") or ""),
        )
    except ValueError as exc:
        return ToolResult(str(exc), is_error=True)
    return ToolResult(f"Saved or refreshed memory #{memory_id}")


async def search_memory(payload: dict[str, Any], context: ToolContext) -> ToolResult:
    if not context.config.features.memory or not context.config.memory.long_term_enabled:
        return ToolResult("Long-term memory is disabled.", is_error=True)
    manager = MemoryManager(context.config.memory.long_term_db_path, scope=context.cwd)
    limit = max(1, min(20, int(payload.get("limit") or 5)))
    memories = manager.search(str(payload["query"]), limit=limit)
    if not memories:
        return ToolResult("No relevant project memories found.")
    lines = [
        f"#{item.id} [{item.category}; importance={item.importance:.1f}; "
        f"confidence={item.confidence:.1f}; key={item.memory_key or '-'}] {item.content}"
        for item in memories
    ]
    return ToolResult(
        "\n".join(lines),
        display_summary=f"Recalled {len(memories)} project memories",
    )


async def load_skill(payload: dict[str, Any], context: ToolContext) -> ToolResult:
    skill = SkillRegistry(context.cwd).load(str(payload["name"]))
    if not skill:
        return ToolResult(f'Skill "{payload["name"]}" not found or disabled.', is_error=True)
    missing = _missing_skill_dependencies(skill, context)
    if missing:
        return ToolResult(
            f'Cannot activate skill "{skill.name}": missing {", ".join(missing)}. '
            "Use available tools or choose another skill.",
            is_error=True,
        )
    content = skill.body or skill.content
    if len(content) > 5_000:
        content = content[:5_000] + "\n... [truncated; use /skill show for the full skill]"
    resources = skill.resource_names()
    if resources:
        content += (
            "\n\nAvailable skill resources (load only when needed with "
            "read_skill_resource):\n- " + "\n- ".join(resources)
        )
    if context.skill_context_buffer:
        context.skill_context_buffer.push(skill.name, content)
        return ToolResult(
            f'Loaded skill "{skill.name}" instructions for the next model turn.',
            display_summary=f"Loaded skill {skill.name}",
        )
    return ToolResult(content, display_summary=f"Loaded skill {skill.name}")


async def read_skill_resource(payload: dict[str, Any], context: ToolContext) -> ToolResult:
    skill = SkillRegistry(context.cwd).load(str(payload["name"]))
    if not skill:
        return ToolResult(f'Skill "{payload["name"]}" not found or disabled.', is_error=True)
    relative = str(payload["path"]).replace("\\", "/").strip()
    if not relative or Path(relative).is_absolute():
        return ToolResult("Skill resource path must be relative.", is_error=True)
    allowed = set(skill.resource_names())
    if relative not in allowed:
        return ToolResult(
            f'Resource "{relative}" is not advertised by skill "{skill.name}".',
            is_error=True,
        )
    root = skill.path.parent.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.is_symlink():
        return ToolResult("Skill resource path is invalid.", is_error=True)
    limit = max(500, min(20_000, int(payload.get("max_chars") or 8_000)))
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return ToolResult(f"Unable to read skill resource: {exc}", is_error=True)
    truncated = len(content) > limit
    if truncated:
        content = content[:limit] + "\n... [skill resource truncated]"
    return ToolResult(
        content,
        display_summary=f"Read {skill.name}/{relative}" + (" (truncated)" if truncated else ""),
    )


def _missing_skill_dependencies(skill, context: ToolContext) -> list[str]:
    registry = context.tool_registry
    if not registry:
        return []
    available = set(registry.list_names())
    missing = [f"tool:{name}" for name in skill.requires_tools if name not in available]
    for server in skill.requires_mcp:
        prefix = f"mcp__{server}__"
        if not any(name.startswith(prefix) for name in available):
            missing.append(f"mcp:{server}")
    return missing


async def search_code(payload: dict[str, Any], context: ToolContext) -> ToolResult:
    try:
        results, truncated = _navigator(context).search(
            str(payload["query"]),
            mode=str(payload.get("mode") or "auto"),
            path=str(payload.get("path") or "."),
            kind=str(payload.get("kind") or ""),
            regex=bool(payload.get("regex", False)),
            limit=int(payload.get("limit") or 20),
        )
    except (RuntimeError, ValueError, re.error) as exc:
        return ToolResult(f"search_code failed: {exc}", is_error=True)
    return ToolResult(
        _search_result_json(
            results,
            truncated=truncated,
            query=str(payload["query"]),
        )
    )


async def repo_map(payload: dict[str, Any], context: ToolContext) -> ToolResult:
    max_chars = max(1000, min(int(payload.get("max_chars") or 12_000), 20_000))
    return ToolResult(_navigator(context).repo_map(max_chars=max_chars))


async def document_symbols(payload: dict[str, Any], context: ToolContext) -> ToolResult:
    path = str(Path(str(payload["path"])))
    symbols = _navigator(context).document_symbols(
        path,
        limit=int(payload.get("limit") or 100),
    )
    return ToolResult(
        json.dumps(
            {
                "path": path,
                "symbols": [
                    {
                        "name": item.name,
                        "kind": item.kind,
                        "parent_name": item.parent_name,
                        "signature": item.signature,
                        "start_line": item.start_line,
                        "end_line": item.end_line,
                    }
                    for item in symbols
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


async def diagnose_source_file(payload: dict[str, Any], context: ToolContext) -> ToolResult:
    path = _resolve_path(context, str(payload["path"]))
    diagnostics = run_diagnostics(path)
    return ToolResult(
        json.dumps(
            {"path": str(path.relative_to(context.cwd)), "diagnostics": diagnostics},
            ensure_ascii=False,
            indent=2,
        ),
        is_error=bool(diagnostics),
    )


def _navigator(context: ToolContext) -> CodeNavigator:
    if context.code_navigator is None:
        context.code_navigator = CodeNavigator(context.cwd)
    return context.code_navigator


def _search_result_json(results, *, truncated: bool, query: str = "") -> str:
    return json.dumps(
        {
            "query": query,
            "matches": [
                {
                    "path": item.path,
                    "start_line": item.start_line,
                    "end_line": item.end_line,
                    "snippet": item.snippet,
                    "symbol": item.symbol,
                    "reason": item.reason,
                    "file_version": item.file_version,
                }
                for item in results
            ],
            "truncated": truncated,
        },
        ensure_ascii=False,
        indent=2,
    )


async def revert_turn(payload: dict[str, Any], context: ToolContext) -> ToolResult:
    record = SnapshotService(context.cwd).restore(str(payload["snapshot"]))
    return ToolResult(f"Restored snapshot {record.id}")


def _resolve_path(context: ToolContext, value: str) -> Path:
    if context.config.policy.path_guard_enabled:
        return PathGuard(context.cwd).validate(value)
    path = Path(value)
    return path if path.is_absolute() else Path(context.cwd).resolve() / path


def _skip_file(path: Path) -> bool:
    skip_dirs = {".git", ".venv", "node_modules", "dist", "build", "target"}
    if any(part in skip_dirs for part in path.parts):
        return True
    return path.stat().st_size > 1_000_000
