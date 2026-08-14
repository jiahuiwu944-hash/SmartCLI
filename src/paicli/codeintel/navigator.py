from __future__ import annotations

import hashlib
import os
import re
import sqlite3
from pathlib import Path

from paicli.codeintel.models import SearchResult, Symbol
from paicli.codeintel.repository import RepositoryScanner
from paicli.codeintel.symbols import LANGUAGES, extract_symbols


class CodeNavigator:
    def __init__(self, root: str | Path, db_path: str | Path | None = None):
        self.root = Path(root).resolve()
        self.db_path = Path(db_path) if db_path else self.root / ".paicli" / "codeintel.sqlite3"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.scanner = RepositoryScanner(self.root)
        self._ensure_schema()

    def update(self, path: str | Path | None = None) -> dict[str, int]:
        files = self.scanner.files(path)
        discovered = {str(item.relative_to(self.root)): item for item in files}
        changed = 0
        skipped = 0
        removed = 0
        with self._connect() as conn:
            existing = {
                row[0]: (int(row[1]), int(row[2]), str(row[3]))
                for row in conn.execute(
                    "select path, size, mtime_ns, sha256 from files where root = ?",
                    (str(self.root),),
                )
            }
            for relative, file_path in discovered.items():
                file_stat = file_path.stat()
                previous = existing.get(relative)
                if previous and previous[:2] == (file_stat.st_size, file_stat.st_mtime_ns):
                    skipped += 1
                    continue
                digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
                if previous and previous[2] == digest:
                    conn.execute(
                        "update files set size = ?, mtime_ns = ? where root = ? and path = ?",
                        (file_stat.st_size, file_stat.st_mtime_ns, str(self.root), relative),
                    )
                    skipped += 1
                    continue
                self._replace_file(conn, file_path, relative, digest)
                changed += 1

            if path is None or str(path) in {"", ".", str(self.root)}:
                for relative in set(existing) - set(discovered):
                    self._delete_file(conn, relative)
                    removed += 1
        return {"changed": changed, "skipped": skipped, "removed": removed}

    def refresh_file(self, path: str | Path) -> None:
        file_path = self._resolve(path)
        relative = str(file_path.relative_to(self.root))
        with self._connect() as conn:
            if not file_path.exists():
                self._delete_file(conn, relative)
                return
            digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
            self._replace_file(conn, file_path, relative, digest)

    def find_symbol(
        self,
        name: str,
        *,
        kind: str = "",
        path: str = ".",
        limit: int = 20,
        refresh: bool = True,
    ) -> list[Symbol]:
        if refresh:
            self.update(path if path != "." else None)
        query = (
            "select path, name, kind, parent_name, signature, start_line, end_line, "
            "docstring, file_sha256 from symbols "
            "where root = ? and lower(name) = lower(?)"
        )
        params: list[object] = [str(self.root), name]
        if kind:
            query += " and lower(kind) = lower(?)"
            params.append(kind)
        path_clause, path_params = self._symbol_path_filter(path)
        query += path_clause
        params.extend(path_params)
        query += " order by path, start_line limit ?"
        params.append(max(1, min(limit, 50)))
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [Symbol(*row) for row in rows]

    def document_symbols(self, path: str, *, limit: int = 100) -> list[Symbol]:
        self.update(path)
        with self._connect() as conn:
            rows = conn.execute(
                """
                select path, name, kind, parent_name, signature, start_line, end_line,
                       docstring, file_sha256
                from symbols where root = ? and path = ?
                order by start_line limit ?
                """,
                (str(self.root), str(Path(path)), max(1, min(limit, 200))),
            ).fetchall()
        return [Symbol(*row) for row in rows]

    def search(
        self,
        query: str,
        *,
        mode: str = "auto",
        path: str = ".",
        kind: str = "",
        regex: bool = False,
        limit: int = 20,
    ) -> tuple[list[SearchResult], bool]:
        query = query.strip()
        if not query:
            raise ValueError("search query must not be empty")
        mode = mode.strip().lower()
        allowed_modes = {"auto", "symbol", "text", "references"}
        if mode not in allowed_modes:
            supported = ", ".join(sorted(allowed_modes))
            raise ValueError(f"unsupported search mode: {mode}; expected one of {supported}")

        capped_limit = max(1, min(limit, 50))
        results: list[SearchResult] = []
        identifier = query.replace(".", "").replace("_", "").isalnum()
        if mode in {"auto", "symbol"} and identifier:
            self.update(path if path != "." else None)
            for symbol in self.find_symbol(
                query,
                kind=kind,
                path=path,
                limit=capped_limit,
                refresh=False,
            ):
                results.append(
                    SearchResult(
                        path=symbol.path,
                        start_line=symbol.start_line,
                        end_line=symbol.end_line,
                        snippet=symbol.signature or symbol.docstring,
                        reason="exact symbol match",
                        symbol=symbol.name,
                        file_version=f"sha256:{symbol.file_version}",
                    )
                )
        truncated = False
        if mode in {"auto", "text"} and len(results) < capped_limit:
            text_results, truncated = self.scanner.search(
                query,
                path=path,
                regex=regex if mode == "text" else False,
                limit=capped_limit - len(results),
            )
            results.extend(text_results)
        if mode == "references":
            reference_results, truncated = self.find_references(
                query,
                path=path,
                limit=capped_limit,
            )
            results.extend(reference_results)
        deduplicated: list[SearchResult] = []
        seen: set[tuple[str, int]] = set()
        for result in results:
            key = (result.path, result.start_line)
            if key not in seen:
                seen.add(key)
                deduplicated.append(result)
        return deduplicated[:capped_limit], truncated or len(deduplicated) > capped_limit

    def find_references(self, name: str, *, path: str = ".", limit: int = 50):
        return self.scanner.search(
            rf"\b{re.escape(name)}\b",
            path=path,
            regex=True,
            limit=limit,
        )

    def repo_map(self, *, max_chars: int = 12_000) -> str:
        self.update()
        lines = [f"Repository: {self.root.name}"]
        with self._connect() as conn:
            files = conn.execute(
                "select path, language from files where root = ? order by path",
                (str(self.root),),
            ).fetchall()
            symbols = conn.execute(
                """
                select path, name, kind, start_line from symbols
                where root = ? order by path, start_line
                """,
                (str(self.root),),
            ).fetchall()
        symbols_by_path: dict[str, list[str]] = {}
        for path, name, kind, line in symbols:
            symbols_by_path.setdefault(str(path), []).append(f"{kind} {name}@{line}")
        for path, language in files:
            entry = f"- {path} [{language or 'text'}]"
            selected = symbols_by_path.get(str(path), [])[:12]
            if selected:
                entry += ": " + ", ".join(selected)
            if sum(len(item) + 1 for item in lines) + len(entry) > max_chars:
                lines.append("... [repo map truncated]")
                break
            lines.append(entry)
        return "\n".join(lines)

    def _replace_file(
        self,
        conn: sqlite3.Connection,
        path: Path,
        relative: str,
        digest: str,
    ) -> None:
        file_stat = path.stat()
        language = LANGUAGES.get(path.suffix.lower(), path.suffix.lower().lstrip("."))
        self._delete_file(conn, relative)
        conn.execute(
            "insert into files(root, path, language, size, mtime_ns, sha256) "
            "values (?, ?, ?, ?, ?, ?)",
            (str(self.root), relative, language, file_stat.st_size, file_stat.st_mtime_ns, digest),
        )
        try:
            symbols = extract_symbols(path, relative_path=relative, file_version=digest)
        except OSError:
            symbols = []
        for symbol in symbols:
            values = (
                str(self.root),
                symbol.path,
                symbol.name,
                symbol.kind,
                symbol.parent_name,
                symbol.signature,
                symbol.start_line,
                symbol.end_line,
                symbol.docstring,
                symbol.file_version,
            )
            conn.execute(
                """
                insert into symbols(root, path, name, kind, parent_name, signature,
                                    start_line, end_line, docstring, file_sha256)
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )

    def _delete_file(self, conn: sqlite3.Connection, relative: str) -> None:
        conn.execute("delete from files where root = ? and path = ?", (str(self.root), relative))
        conn.execute("delete from symbols where root = ? and path = ?", (str(self.root), relative))

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                create table if not exists files (
                    root text not null, path text not null, language text not null,
                    size integer not null, mtime_ns integer not null, sha256 text not null,
                    primary key(root, path)
                );
                create table if not exists symbols (
                    id integer primary key autoincrement, root text not null, path text not null,
                    name text not null, kind text not null, parent_name text not null,
                    signature text not null, start_line integer not null, end_line integer not null,
                    docstring text not null, file_sha256 text not null
                );
                create index if not exists idx_symbols_root_name on symbols(root, name);
                create index if not exists idx_symbols_root_path on symbols(root, path);
                """
            )
            legacy_fts = conn.execute(
                "select 1 from sqlite_master where type = 'table' and name = 'symbols_fts'"
            ).fetchone()
            if legacy_fts:
                conn.execute("drop table symbols_fts")

    def _symbol_path_filter(self, path: str) -> tuple[str, list[object]]:
        if path in {"", "."}:
            return "", []
        resolved = self._resolve(path)
        if resolved == self.root:
            return "", []
        relative = str(resolved.relative_to(self.root))
        if resolved.is_file():
            return " and path = ?", [relative]
        prefix = relative.rstrip("\\/") + os.sep
        return " and (path = ? or substr(path, 1, ?) = ?)", [relative, len(prefix), prefix]

    def _resolve(self, value: str | Path) -> Path:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        resolved = candidate.resolve()
        resolved.relative_to(self.root)
        return resolved

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)
