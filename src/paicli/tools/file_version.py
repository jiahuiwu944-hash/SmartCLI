from __future__ import annotations

import hashlib
import os
import stat
import tempfile
from pathlib import Path

MISSING_VERSION = "missing"


class FileChangedDuringWriteError(RuntimeError):
    def __init__(self, actual_version: str):
        super().__init__("file changed while the replacement was being prepared")
        self.actual_version = actual_version


def file_version(path: Path) -> str:
    if not path.exists():
        return MISSING_VERSION
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"sha256:{digest}"


def content_version(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def atomic_write(path: Path, content: bytes, *, observed_version: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    original_mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else None
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if original_mode is not None:
            os.chmod(temporary_path, original_mode)

        actual_version = file_version(path)
        if actual_version != observed_version:
            raise FileChangedDuringWriteError(actual_version)

        if observed_version == MISSING_VERSION:
            try:
                os.link(temporary_path, path)
            except FileExistsError as exc:
                raise FileChangedDuringWriteError(file_version(path)) from exc
            temporary_path.unlink()
        else:
            os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
