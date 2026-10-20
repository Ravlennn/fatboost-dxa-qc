"""Bounded ZIP extraction and stable logical paths for batch inputs."""
from __future__ import annotations

import os
import shutil
import stat
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

MAX_FILES = 200_000                 # far above any test batch; guards against zip bombs only
MAX_ARCHIVE_BYTES = 64 * 1024 ** 3
MAX_FILE_BYTES = 512 * 1024 ** 2
FREE_SPACE_MARGIN = 1024 ** 3
MAX_NESTING = 2
UNICODE_PATH_FIELD = 0x7075


def _has_unicode_path_field(extra: bytes) -> bool:
    i = 0
    while i + 4 <= len(extra):
        header, size = int.from_bytes(extra[i:i + 2], 'little'), int.from_bytes(extra[i + 2:i + 4], 'little')
        if header == UNICODE_PATH_FIELD:
            return True
        i += 4 + size
    return False


def entry_name(entry: zipfile.ZipInfo) -> str:
    """ZIP member name with its encoding restored.

    Without the UTF-8 flag and without the Unicode Path field (0x7075) Python decodes the name as
    cp437, so Russian names turn into mojibake and path_to_study no longer matches the organizer's
    reference. Such names are re-read as UTF-8 (macOS, Linux archivers), then cp866 (Windows).
    Archives with the flag or the field are left alone (the organizer's archives carry the field).
    """
    name = entry.filename
    if entry.flag_bits & 0x800 or name.isascii() or _has_unicode_path_field(entry.extra):
        return name
    try:
        raw = name.encode('cp437')
    except UnicodeEncodeError:
        return name
    for encoding in ('utf-8', 'cp866'):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return name


def _safe_name(raw: str) -> PurePosixPath | None:
    """Normalised relative path, or None if it would escape the extraction folder.

    Windows archivers may store '\\' separators; they are turned into '/' before the checks.
    """
    name = raw.replace('\\', '/')
    while name.startswith('./'):
        name = name[2:]
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or '..' in path.parts or ':' in name
            or any(part in ('', '.') for part in path.parts)):
        return None
    return path


def _extract(zip_path: Path, dest: Path, skipped: list, label: str = '', depth: int = 0) -> None:
    """Extract one ZIP into dest; bad members are skipped and recorded, nested ZIPs expanded."""
    with zipfile.ZipFile(zip_path) as archive:
        entries = archive.infolist()
        total = sum(x.file_size for x in entries)
        if len(entries) > MAX_FILES or total > MAX_ARCHIVE_BYTES:
            raise ValueError('ZIP exceeds file count or uncompressed size limit')
        if shutil.disk_usage(dest).free < total + FREE_SPACE_MARGIN:
            raise OSError(f'Not enough free disk space to extract {total / 1024 ** 3:.1f} GiB')
        taken: dict[str, int] = {}
        inner = []
        for entry in entries:
            if entry.is_dir():
                continue
            fixed = entry_name(entry)
            shown = label + fixed
            name = _safe_name(fixed)
            mode = entry.external_attr >> 16
            reason = ('unsafe path' if name is None else 'symbolic link' if stat.S_ISLNK(mode)
                      else 'encrypted' if entry.flag_bits & 1
                      else 'member exceeds size limit' if entry.file_size > MAX_FILE_BYTES else None)
            if reason:
                skipped.append({'path': shown, 'reason': reason})
                continue
            key = name.as_posix().casefold()
            if key in taken:
                # the same name again, or one differing only in case (case-insensitive disks):
                # keep it under a distinct name instead of overwriting the first
                taken[key] += 1
                name = name.with_name(f'{name.stem}__dup{taken[key]}{name.suffix}')
                skipped.append({'path': shown, 'reason': f'duplicate name, stored as {name.as_posix()}', 'kept': True})
            else:
                taken[key] = 1
            target = dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(entry) as incoming, target.open('wb') as outgoing:
                shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
            if target.suffix.lower() == '.zip':
                inner.append((target, shown))
    for target, shown in inner:
        _expand_inner(target, skipped, shown, depth)


def _expand_inner(target: Path, skipped: list, shown: str, depth: int) -> None:
    """A ZIP inside the input becomes a folder of the same name (without .zip), up to 2 levels."""
    if depth >= MAX_NESTING:
        skipped.append({'path': shown, 'reason': 'nested ZIP deeper than 2 levels'})
        target.unlink()
        return
    folder = target.with_suffix('')
    while folder.exists():
        folder = folder.with_name(folder.name + '_zip')
    folder.mkdir(parents=True)
    try:
        _extract(target, folder, skipped, label=shown + '/', depth=depth + 1)
    except (zipfile.BadZipFile, ValueError, OSError) as error:
        skipped.append({'path': shown, 'reason': f'nested ZIP not readable: {error}'})
    target.unlink()


@contextmanager
def input_directory(source: Path, skipped: list | None = None):
    """Yield a folder with the input files; ZIPs (the input itself or inside it) are extracted.

    One bad archive member never stops the batch: unsafe paths, symlinks, encrypted or oversized
    members are left unextracted and appended to `skipped` as {'path', 'reason'} for the report.
    Nested ZIPs (up to two levels) and ZIPs inside an input folder are expanded in place of the
    archive. Only the zip-bomb limits and missing disk space raise.
    """
    skipped = skipped if skipped is not None else []
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f'Input does not exist: {source}')
    if source.is_dir():
        zips = [p for p in source.rglob('*') if p.is_file() and p.suffix.lower() == '.zip']
        if not zips:
            yield source
            return
        # a folder that contains archives: mirror it into a temporary folder, archives expanded
        with tempfile.TemporaryDirectory(prefix='dxaqc-') as directory:
            root = Path(directory)
            for p in sorted(source.rglob('*')):
                if not p.is_file():
                    continue
                target = root / p.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(p, target)
                except OSError:
                    shutil.copy2(p, target)
                if target.suffix.lower() == '.zip':
                    _expand_inner(target, skipped, p.relative_to(source).as_posix(), 0)
            yield root
        return
    if source.suffix.lower() != '.zip':
        yield source
        return
    with tempfile.TemporaryDirectory(prefix='dxaqc-') as directory:
        root = Path(directory)
        _extract(source, root, skipped)
        yield root
