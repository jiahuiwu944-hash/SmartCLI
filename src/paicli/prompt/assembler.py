from __future__ import annotations

from datetime import datetime
from pathlib import Path

from paicli.config import PaiCliConfig
from paicli.memory import MemoryManager
from paicli.skill import SkillRegistry


class PromptAssembler:
    def __init__(
        self,
        config: PaiCliConfig,
        cwd: str,
        tool_names: list[str],
        model: str,
        provider: str,
    ):
        self.config = config
        self.cwd = str(Path(cwd).resolve())
        self.tool_names = tool_names
        self.model = model
        self.provider = provider

    def build(self) -> str:
        parts = [
            "You are SmartCLI, a powerful AI coding assistant running in a terminal.",
            f"Current time: {datetime.now().isoformat(timespec='seconds')}",
            f"Working directory: {self.cwd}",
            f"Model: {self.model} ({self.provider})",
            f"Available tools: {', '.join(self.tool_names)}",
            "",
            "Guidelines:",
            "- Be concise, direct, and implementation-oriented.",
            "- Use tools to inspect files, search code, and verify behavior when needed.",
            "- Prefer deterministic local tools before guessing.",
            "- If an exact file path is known, call read_file directly; do not search the "
            "whole repository.",
            "- If the project structure is unknown, use repo_map to identify likely modules; "
            "do not call it when an exact path or search scope is already known.",
            "- Use search_code as the unified code search entry: symbol for exact definitions, "
            "text for literal or regex matches, references for possible usages, and auto when "
            "the best strategy is unclear.",
            "- If an exact file path is known but its internal structure is unclear, use "
            "document_symbols before reading only the relevant source range.",
            "- Treat search_code, repo_map, and document_symbols results as candidates; call "
            "read_file to verify the current source before making changes.",
            "- Search results and the repo map are navigation hints, not authoritative "
            "file content.",
            "- If a search returns nothing, change the keyword, symbol, search mode, or "
            "reference direction instead of repeating it.",
            "- Before changing code, inspect the target symbol, its callers, and relevant "
            "tests; after changing it, run diagnose_file or tests.",
            "- When writing files, use write_file and keep changes scoped.",
            "- Before overwriting or appending to an existing file, call read_file and pass "
            "its returned version to write_file as expected_version.",
            "- If write_file returns FILE_VERSION_CONFLICT, read the file again and rebuild "
            "the change from the latest content; never guess a version or force an overwrite.",
            "- Preserve URLs and user-provided identifiers exactly unless a tool result proves "
            "otherwise.",
            "- Ask a clarifying question only when proceeding would be risky.",
        ]
        project_memory = self._project_memory()
        if project_memory:
            parts.extend(["", "Project memory:", project_memory])
        skill_index = SkillRegistry(self.cwd).index_text() if self.config.features.skill else ""
        if skill_index:
            parts.extend(["", skill_index])
        return "\n".join(parts)

    def _project_memory(self) -> str:
        memory_files = [
            Path(self.cwd) / "PAI.md",
            Path(self.cwd) / ".paicli" / "PAI.md",
            Path(self.cwd) / "PAI.local.md",
            Path(self.cwd) / ".paicli" / "PAI.local.md",
        ]
        chunks = []
        for path in memory_files:
            if path.exists():
                try:
                    chunks.append(path.read_text(encoding="utf-8")[:4000])
                except OSError:
                    continue
        if self.config.features.memory and self.config.memory.long_term_enabled:
            manager = MemoryManager(self.config.memory.long_term_db_path, scope=self.cwd)
            memories = manager.list(limit=8)
            if memories:
                chunks.append("\n".join(f"- {item.content}" for item in memories))
        return "\n\n".join(chunks)[:8000]
