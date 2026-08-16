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

    def covers(
        self,
        path: str,
        start_line: int,
        end_line: int,
        file_version: str,
    ) -> bool:
        """Return whether recorded regions cover the requested range without gaps."""

        if end_line < start_line:
            return True
        regions = sorted(
            (
                region
                for region in self._regions
                if region.path == path and region.file_version == file_version
            ),
            key=lambda region: (region.start_line, region.end_line),
        )
        covered_until = start_line - 1
        for region in regions:
            if region.end_line < start_line:
                continue
            if region.start_line > covered_until + 1:
                return False
            covered_until = max(covered_until, region.end_line)
            if covered_until >= end_line:
                return True
        return False

    def clear(self) -> None:
        self._regions.clear()
