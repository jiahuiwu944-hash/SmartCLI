from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Symbol:
    path: str
    name: str
    kind: str
    parent_name: str
    signature: str
    start_line: int
    end_line: int
    docstring: str
    file_version: str


@dataclass(slots=True)
class SearchResult:
    path: str
    start_line: int
    end_line: int
    snippet: str
    reason: str
    symbol: str = ""
    file_version: str = ""
