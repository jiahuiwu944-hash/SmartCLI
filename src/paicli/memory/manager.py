from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

_VALID_CATEGORIES = {"fact", "preference", "project", "decision", "constraint", "solution"}
_QUERY_STOP_WORDS = {
    "a",
    "an",
    "for",
    "please",
    "the",
    "this",
    "to",
    "user",
    "with",
    "一个",
    "一下",
    "什么",
    "如何",
    "怎么",
    "我们",
    "现在",
    "这个",
    "那个",
    "可以",
    "需要",
}


@dataclass(slots=True)
class MemoryEntry:
    id: int
    scope: str
    content: str
    created_at: str
    category: str = "fact"
    importance: float = 0.5
    source: str = "agent"
    updated_at: str = ""
    last_accessed_at: str | None = None
    access_count: int = 0
    memory_key: str = ""
    confidence: float = 0.5
    evidence: str = ""
    status: str = "active"
    superseded_by: int | None = None
    score: float = 0.0


@dataclass(slots=True)
class MemoryMutation:
    memory_id: int
    action: str
    previous_id: int | None = None


@dataclass(slots=True)
class MemoryEvent:
    id: int
    memory_id: int
    action: str
    details: str
    created_at: str


class MemoryManager:
    """Project-scoped long-term memory with deduplication and ranked recall."""

    def __init__(self, db_path: str | Path, scope: str):
        self.db_path = Path(db_path).expanduser()
        self.scope = str(Path(scope).resolve())
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def save(
        self,
        content: str,
        *,
        category: str = "fact",
        importance: float = 0.5,
        source: str = "agent",
        memory_key: str = "",
        confidence: float = 0.5,
        evidence: str = "",
    ) -> int:
        return self.upsert(
            content,
            category=category,
            importance=importance,
            source=source,
            memory_key=memory_key,
            confidence=confidence,
            evidence=evidence,
        ).memory_id

    def upsert(
        self,
        content: str,
        *,
        category: str = "fact",
        importance: float = 0.5,
        source: str = "agent",
        memory_key: str = "",
        confidence: float = 0.5,
        evidence: str = "",
    ) -> MemoryMutation:
        normalized = _normalize_content(content)
        if not normalized:
            raise ValueError("Memory content cannot be empty.")
        if _contains_sensitive_value(normalized):
            raise ValueError("Refusing to store a possible secret in long-term memory.")
        category = _normalize_category(category)
        importance = _clamp_importance(importance)
        confidence = _clamp_importance(confidence)
        memory_key = _normalize_key(memory_key)
        evidence = _normalize_content(evidence)[:2000]
        now = datetime.now(UTC).isoformat()
        content_hash = _content_hash(normalized)
        with self._connect() as conn:
            conn.execute("begin immediate")
            existing = None
            if memory_key:
                existing = conn.execute(
                    """
                    select id, importance, content_hash, memory_key, confidence, evidence
                    from memories
                    where scope = ? and memory_key = ? and status = 'active'
                    order by id desc
                    limit 1
                    """,
                    (self.scope, memory_key),
                ).fetchone()
            if existing is None and not memory_key:
                existing = conn.execute(
                    """
                    select id, importance, content_hash, memory_key, confidence, evidence
                    from memories
                    where scope = ? and content_hash = ? and status = 'active'
                    order by id desc
                    limit 1
                    """,
                    (self.scope, content_hash),
                ).fetchone()
            if existing:
                memory_id = int(existing[0])
                if str(existing[2] or "") != content_hash and memory_key:
                    conn.execute(
                        """
                        update memories
                        set status = 'superseded', updated_at = ?
                        where id = ?
                        """,
                        (now, memory_id),
                    )
                    cursor = conn.execute(
                        """
                        insert into memories(
                            scope, content, created_at, category, importance, source,
                            updated_at, content_hash, memory_key, confidence, evidence, status
                        ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active')
                        """,
                        (
                            self.scope,
                            normalized,
                            now,
                            category,
                            importance,
                            source,
                            now,
                            content_hash,
                            memory_key,
                            confidence,
                            evidence,
                        ),
                    )
                    replacement_id = int(cursor.lastrowid)
                    conn.execute(
                        """
                        update memories
                        set superseded_by = ?, updated_at = ?
                        where id = ?
                        """,
                        (replacement_id, now, memory_id),
                    )
                    self._record_event(
                        conn,
                        memory_id,
                        "superseded",
                        {"replacement_id": replacement_id, "memory_key": memory_key},
                        now,
                    )
                    self._record_event(
                        conn,
                        replacement_id,
                        "inserted",
                        {"previous_id": memory_id, "memory_key": memory_key},
                        now,
                    )
                    return MemoryMutation(replacement_id, "replaced", memory_id)
                conn.execute(
                    """
                    update memories set
                        content = ?, category = ?, importance = ?, source = ?, updated_at = ?,
                        memory_key = ?, confidence = ?, evidence = ?
                    where id = ?
                    """,
                    (
                        normalized,
                        category,
                        max(float(existing[1] or 0.0), importance),
                        source,
                        now,
                        memory_key or str(existing[3] or ""),
                        max(float(existing[4] or 0.0), confidence),
                        evidence or str(existing[5] or ""),
                        memory_id,
                    ),
                )
                self._record_event(conn, memory_id, "refreshed", {}, now)
                return MemoryMutation(memory_id, "refreshed")
            cursor = conn.execute(
                """
                insert into memories(
                    scope, content, created_at, category, importance, source,
                    updated_at, content_hash, memory_key, confidence, evidence, status
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active')
                """,
                (
                    self.scope,
                    normalized,
                    now,
                    category,
                    importance,
                    source,
                    now,
                    content_hash,
                    memory_key,
                    confidence,
                    evidence,
                ),
            )
            memory_id = int(cursor.lastrowid)
            self._record_event(
                conn,
                memory_id,
                "inserted",
                {"memory_key": memory_key},
                now,
            )
            return MemoryMutation(memory_id, "inserted")

    def list(self, limit: int = 20, *, include_inactive: bool = False) -> list[MemoryEntry]:
        status_clause = "" if include_inactive else "and status = 'active'"
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                select {_SELECT_COLUMNS}
                from memories
                where scope = ? {status_clause}
                order by updated_at desc, id desc
                limit ?
                """,
                (self.scope, max(1, int(limit))),
            ).fetchall()
        return [_entry(row) for row in rows]

    def search(self, query: str, limit: int = 10) -> list[MemoryEntry]:
        return self.recall(query, limit=limit, min_score=0.01, record_access=False)

    def recall(
        self,
        query: str,
        *,
        limit: int = 5,
        min_score: float = 1.0,
        record_access: bool = True,
    ) -> list[MemoryEntry]:
        query = _normalize_content(query).lower()
        if not query:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                select {_SELECT_COLUMNS}
                from memories
                where scope = ? and status = 'active'
                order by updated_at desc, id desc
                limit 500
                """,
                (self.scope,),
            ).fetchall()
            ranked: list[MemoryEntry] = []
            now = datetime.now(UTC)
            for row in rows:
                item = _entry(row)
                item.score = _memory_score(query, item, now)
                if item.score >= float(min_score):
                    ranked.append(item)
            ranked.sort(key=lambda item: (item.score, item.importance, item.id), reverse=True)
            selected = ranked[: max(1, int(limit))]
            if record_access and selected:
                accessed_at = now.isoformat()
                conn.executemany(
                    """
                    update memories
                    set last_accessed_at = ?, access_count = access_count + 1
                    where id = ? and scope = ?
                    """,
                    [(accessed_at, item.id, self.scope) for item in selected],
                )
                for item in selected:
                    item.last_accessed_at = accessed_at
                    item.access_count += 1
        return selected

    def delete(self, memory_id: int) -> bool:
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                update memories set status = 'deleted', updated_at = ?
                where id = ? and scope = ? and status != 'deleted'
                """,
                (now, int(memory_id), self.scope),
            )
            if cursor.rowcount:
                self._record_event(conn, int(memory_id), "deleted", {}, now)
            return bool(cursor.rowcount)

    def restore(self, memory_id: int) -> bool:
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            row = conn.execute(
                "select memory_key from memories where id = ? and scope = ?",
                (int(memory_id), self.scope),
            ).fetchone()
            if row is None:
                return False
            memory_key = str(row[0] or "")
            replaced_ids: list[int] = []
            if memory_key:
                replaced_ids = [
                    int(item[0])
                    for item in conn.execute(
                        """
                        select id from memories
                        where scope = ? and memory_key = ? and status = 'active' and id != ?
                        """,
                        (self.scope, memory_key, int(memory_id)),
                    ).fetchall()
                ]
                conn.execute(
                    """
                    update memories set status = 'superseded', superseded_by = ?, updated_at = ?
                    where scope = ? and memory_key = ? and status = 'active' and id != ?
                    """,
                    (int(memory_id), now, self.scope, memory_key, int(memory_id)),
                )
            cursor = conn.execute(
                """
                update memories set status = 'active', superseded_by = null, updated_at = ?
                where id = ? and scope = ?
                """,
                (now, int(memory_id), self.scope),
            )
            if cursor.rowcount:
                for replaced_id in replaced_ids:
                    self._record_event(
                        conn,
                        replaced_id,
                        "superseded",
                        {"replacement_id": int(memory_id), "reason": "restore"},
                        now,
                    )
                self._record_event(
                    conn,
                    int(memory_id),
                    "restored",
                    {"replaced_ids": replaced_ids},
                    now,
                )
            return bool(cursor.rowcount)

    def audit(self, limit: int = 20) -> list[MemoryEvent]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                select id, memory_id, action, details, created_at
                from memory_events
                where scope = ?
                order by id desc
                limit ?
                """,
                (self.scope, max(1, int(limit))),
            ).fetchall()
        return [MemoryEvent(*row) for row in rows]

    def clear(self) -> int:
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            ids = [
                int(row[0])
                for row in conn.execute(
                    "select id from memories where scope = ? and status = 'active'",
                    (self.scope,),
                ).fetchall()
            ]
            cursor = conn.execute(
                """
                update memories set status = 'deleted', updated_at = ?
                where scope = ? and status = 'active'
                """,
                (now, self.scope),
            )
            for memory_id in ids:
                self._record_event(conn, memory_id, "deleted", {"reason": "clear"}, now)
            return int(cursor.rowcount)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                create table if not exists memories (
                    id integer primary key autoincrement,
                    scope text not null,
                    content text not null,
                    created_at text not null,
                    category text not null default 'fact',
                    importance real not null default 0.5,
                    source text not null default 'agent',
                    updated_at text not null default '',
                    last_accessed_at text,
                    access_count integer not null default 0,
                    content_hash text not null default '',
                    memory_key text not null default '',
                    confidence real not null default 0.5,
                    evidence text not null default '',
                    status text not null default 'active',
                    superseded_by integer
                )
                """
            )
            columns = {
                str(row[1])
                for row in conn.execute("pragma table_info(memories)").fetchall()
            }
            migrations = {
                "category": "text not null default 'fact'",
                "importance": "real not null default 0.5",
                "source": "text not null default 'agent'",
                "updated_at": "text not null default ''",
                "last_accessed_at": "text",
                "access_count": "integer not null default 0",
                "content_hash": "text not null default ''",
                "memory_key": "text not null default ''",
                "confidence": "real not null default 0.5",
                "evidence": "text not null default ''",
                "status": "text not null default 'active'",
                "superseded_by": "integer",
            }
            for name, declaration in migrations.items():
                if name not in columns:
                    conn.execute(f"alter table memories add column {name} {declaration}")
            rows = conn.execute(
                "select id, content, created_at, updated_at, content_hash from memories"
            ).fetchall()
            for memory_id, content, created_at, updated_at, content_hash in rows:
                if updated_at and content_hash:
                    continue
                conn.execute(
                    "update memories set updated_at = ?, content_hash = ? where id = ?",
                    (
                        updated_at or created_at,
                        content_hash or _content_hash(_normalize_content(str(content))),
                        memory_id,
                    ),
                )
            conn.execute(
                "create index if not exists idx_memories_scope on memories(scope, id)"
            )
            conn.execute(
                "create index if not exists idx_memories_hash on memories(scope, content_hash)"
            )
            conn.execute(
                """
                create index if not exists idx_memories_key
                on memories(scope, memory_key, status)
                """
            )
            duplicate_keys = conn.execute(
                """
                select scope, memory_key
                from memories
                where status = 'active' and memory_key != ''
                group by scope, memory_key
                having count(*) > 1
                """
            ).fetchall()
            now = datetime.now(UTC).isoformat()
            for scope, memory_key in duplicate_keys:
                ids = [
                    int(row[0])
                    for row in conn.execute(
                        """
                        select id from memories
                        where scope = ? and memory_key = ? and status = 'active'
                        order by id desc
                        """,
                        (scope, memory_key),
                    ).fetchall()
                ]
                newest_id = ids[0]
                for old_id in ids[1:]:
                    conn.execute(
                        """
                        update memories
                        set status = 'superseded', superseded_by = ?, updated_at = ?
                        where id = ?
                        """,
                        (newest_id, now, old_id),
                    )
            conn.execute(
                """
                create unique index if not exists idx_memories_one_active_key
                on memories(scope, memory_key)
                where status = 'active' and memory_key != ''
                """
            )
            conn.execute(
                """
                create table if not exists memory_events (
                    id integer primary key autoincrement,
                    scope text not null,
                    memory_id integer not null,
                    action text not null,
                    details text not null default '{}',
                    created_at text not null
                )
                """
            )
            conn.execute(
                """
                create index if not exists idx_memory_events_scope
                on memory_events(scope, id)
                """
            )

    def _record_event(
        self,
        conn: sqlite3.Connection,
        memory_id: int,
        action: str,
        details: dict,
        created_at: str,
    ) -> None:
        conn.execute(
            """
            insert into memory_events(scope, memory_id, action, details, created_at)
            values (?, ?, ?, ?, ?)
            """,
            (
                self.scope,
                int(memory_id),
                action,
                json.dumps(details, ensure_ascii=False, sort_keys=True),
                created_at,
            ),
        )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)


_SELECT_COLUMNS = (
    "id, scope, content, created_at, category, importance, source, "
    "updated_at, last_accessed_at, access_count, memory_key, confidence, evidence, "
    "status, superseded_by"
)


def _entry(row: tuple) -> MemoryEntry:
    return MemoryEntry(
        id=int(row[0]),
        scope=str(row[1]),
        content=str(row[2]),
        created_at=str(row[3]),
        category=str(row[4] or "fact"),
        importance=float(0.5 if row[5] is None else row[5]),
        source=str(row[6] or "agent"),
        updated_at=str(row[7] or row[3]),
        last_accessed_at=str(row[8]) if row[8] else None,
        access_count=int(row[9] or 0),
        memory_key=str(row[10] or ""),
        confidence=float(0.5 if row[11] is None else row[11]),
        evidence=str(row[12] or ""),
        status=str(row[13] or "active"),
        superseded_by=int(row[14]) if row[14] is not None else None,
    )


def _normalize_content(content: str) -> str:
    return re.sub(r"\s+", " ", str(content)).strip()


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.casefold().encode("utf-8")).hexdigest()


def _contains_sensitive_value(content: str) -> bool:
    patterns = (
        r"\bsk-[a-z0-9_-]{12,}\b",
        r"(?i)\b(?:api[_ -]?key|secret|password|access[_ -]?token)\b\s*[:=]\s*\S{8,}",
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    )
    return any(re.search(pattern, content, flags=re.IGNORECASE) for pattern in patterns)


def _normalize_category(category: str) -> str:
    normalized = str(category or "fact").strip().lower()
    return normalized if normalized in _VALID_CATEGORIES else "fact"


def _normalize_key(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9_.-]+", ".", str(value or "").strip().lower())
    return normalized.strip(".")[:120]


def _clamp_importance(value: float) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.5


def _lexical_units(value: str) -> set[str]:
    units: set[str] = set()
    for token in re.findall(r"[a-z_][a-z0-9_.-]*|\d+|[\u4e00-\u9fff]+", value.lower()):
        token = token.strip(".-")
        if token in _QUERY_STOP_WORDS:
            continue
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            if len(token) <= 8:
                units.add(token)
            units.update(token[index : index + 2] for index in range(len(token) - 1))
        else:
            units.add(token)
    return {unit for unit in units if unit and unit not in _QUERY_STOP_WORDS}


def _memory_score(query: str, item: MemoryEntry, now: datetime) -> float:
    content = item.content.lower()
    query_units = _lexical_units(query)
    content_units = _lexical_units(content)
    matched = query_units & content_units
    exact = query in content or content in query
    if not exact and not matched:
        return 0.0
    coverage = len(matched) / max(1, len(query_units))
    score = (6.0 if exact else 0.0) + coverage * 5.0 + min(len(matched), 6) * 0.25
    score += item.importance * 1.25
    score += item.confidence * 0.75
    try:
        updated = datetime.fromisoformat(item.updated_at or item.created_at)
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=UTC)
        age_days = max(0.0, (now - updated).total_seconds() / 86_400)
        score += 0.5 * math.exp(-age_days / 90.0)
    except (TypeError, ValueError):
        pass
    score += min(item.access_count, 10) * 0.03
    return round(score, 4)
