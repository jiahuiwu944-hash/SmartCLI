from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ReadRegion:
    path: str
    start_line: int
    end_line: int
    file_version: str


class ContextLedger:
    def __init__(self):
        self._regions: set[ReadRegion] = set()

    def seen(self, path: str, start_line: int, end_line: int, file_version: str) -> bool:
        return ReadRegion(path, start_line, end_line, file_version) in self._regions

    def record(self, path: str, start_line: int, end_line: int, file_version: str) -> None:
        self._regions.add(ReadRegion(path, start_line, end_line, file_version))

    def clear(self) -> None:
        self._regions.clear()
