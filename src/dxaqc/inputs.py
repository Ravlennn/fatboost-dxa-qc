"""Bounded ZIP extraction and stable logical paths for batch inputs."""
from __future__ import annotations

import shutil
import stat
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

MAX_FILES = 10000
MAX_ARCHIVE_BYTES = 2 * 1024 ** 3
MAX_FILE_BYTES = 512 * 1024 ** 2


@contextmanager
def input_directory(source: Path):
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f'Input does not exist: {source}')
    if not source.is_file() or source.suffix.lower() != '.zip':
        yield source
        return
    with tempfile.TemporaryDirectory(prefix='dxaqc-') as directory:
        root = Path(directory)
        with zipfile.ZipFile(source) as archive:
            entries = archive.infolist()
            if len(entries) > MAX_FILES or sum(x.file_size for x in entries) > MAX_ARCHIVE_BYTES:
                raise ValueError('ZIP exceeds file count or uncompressed size limit')
            names = set()
            for entry in entries:
                name = PurePosixPath(entry.filename)
                mode = entry.external_attr >> 16
                if (not entry.filename or '\\' in entry.filename or name.is_absolute()
                        or '..' in name.parts or ':' in entry.filename
                        or stat.S_ISLNK(mode) or entry.flag_bits & 1):
                    raise ValueError(f'Unsafe or encrypted ZIP entry: {entry.filename}')
                key = name.as_posix().casefold().rstrip('/')
                if key in names:
                    raise ValueError(f'Duplicate ZIP entry: {entry.filename}')
                names.add(key)
                if entry.file_size > MAX_FILE_BYTES:
                    raise ValueError('ZIP member exceeds file size limit')
            for entry in entries:
                target = root / entry.filename
                if entry.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(entry) as incoming, target.open('wb') as outgoing:
                    shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
        yield root
