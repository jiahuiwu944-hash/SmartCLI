from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from paicli.codeintel.models import SearchResult
from paicli.rag.code_index import SKIP_DIRS as LEGACY_SKIP_DIRS
from paicli.rag.code_index import TEXT_SUFFIXES

SKIP_DIRS = {*LEGACY_SKIP_DIRS, ".paicli"}


class RepositoryScanner:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def files(self, path: str | Path | None = None) -> list[Path]:
        base = self._resolve(path or ".")
        if base.is_file():
            return [base] if self._supported(base) else []
        if shutil.which("rg"):
            relative_base = str(base.relative_to(self.root)) or "."
            command = ["rg", "--files", "--hidden", *self._ignore_args(), relative_base]
            completed = subprocess.run(
                command,
                cwd=self.root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            if completed.returncode in {0, 1}:
                return [
                    candidate
                    for line in completed.stdout.splitlines()
                    if (candidate := self._resolve(line)).is_file() and self._supported(candidate)
                ]
        return [
            candidate
            for candidate in base.rglob("*")
            if candidate.is_file()
            and not any(part in SKIP_DIRS for part in candidate.relative_to(self.root).parts)
            and self._supported(candidate)
        ]

    def search(
        self,
        pattern: str,
        *,
        path: str = ".",
        regex: bool = False,
        limit: int = 50,
        max_chars: int = 12_000,
    ) -> tuple[list[SearchResult], bool]:
        if not shutil.which("rg"):
            return self._fallback_search(
                pattern,
                path=path,
                regex=regex,
                limit=limit,
                max_chars=max_chars,
            )
        command = ["rg", "--json", "--line-number", "--hidden", *self._ignore_args()]
        if not regex:
            command.append("--fixed-strings")
        command.extend([pattern, path])
        completed = subprocess.run(
            command,
            cwd=self.root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if completed.returncode not in {0, 1}:
            raise RuntimeError(completed.stderr.strip() or "ripgrep search failed")
        results: list[SearchResult] = []
        total_chars = 0
        truncated = False
        for line in completed.stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") != "match":
                continue
            data = event.get("data") or {}
            path_text = str((data.get("path") or {}).get("text") or "")
            path_text = str(Path(path_text))
            if path_text.startswith(f".{Path().anchor or '/'}") or path_text.startswith(".\\"):
                path_text = path_text[2:]
            snippet = str((data.get("lines") or {}).get("text") or "").strip()
            line_number = int(data.get("line_number") or 0)
            result = SearchResult(
                path=path_text,
                start_line=line_number,
                end_line=line_number,
                snippet=snippet,
                reason="ripgrep text match",
            )
            result_chars = len(path_text) + len(snippet)
            if len(results) >= max(1, min(limit, 50)) or total_chars + result_chars > max_chars:
                truncated = True
                break
            results.append(result)
            total_chars += result_chars
        return results, truncated

    def _fallback_search(
        self,
        pattern: str,
        *,
        path: str,
        regex: bool,
        limit: int,
        max_chars: int,
    ) -> tuple[list[SearchResult], bool]:
        import re

        compiled = re.compile(pattern) if regex else None
        results: list[SearchResult] = []
        total_chars = 0
        for file_path in self.files(path):
            try:
                lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            except OSError:
                continue
            for line_number, line in enumerate(lines, 1):
                matched = bool(compiled.search(line)) if compiled else pattern in line
                if not matched:
                    continue
                relative = str(file_path.relative_to(self.root))
                snippet = line.strip()
                if len(results) >= max(1, min(limit, 50)) or (
                    total_chars + len(relative) + len(snippet) > max_chars
                ):
                    return results, True
                results.append(
                    SearchResult(
                        path=relative,
                        start_line=line_number,
                        end_line=line_number,
                        snippet=snippet,
                        reason="fallback text match",
                    )
                )
                total_chars += len(relative) + len(snippet)
        return results, False

    @staticmethod
    def _exclude_globs() -> list[str]:
        values: list[str] = []
        for name in sorted(SKIP_DIRS):
            values.extend(["--glob", f"!{name}/**"])
        return values

    def _ignore_args(self) -> list[str]:
        values = self._exclude_globs()
        ignore_file = self.root / ".gitignore"
        if ignore_file.is_file():
            values.extend(["--ignore-file", str(ignore_file)])
        return values

    @staticmethod
    def _supported(path: Path) -> bool:
        return path.suffix.lower() in TEXT_SUFFIXES

    def _resolve(self, value: str | Path) -> Path:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        resolved = candidate.resolve()
        resolved.relative_to(self.root)
        return resolved
