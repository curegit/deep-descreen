import os
import os.path
import glob
import shutil
from pathlib import Path
from typing import IO


def mkdirp(path: str | Path, recreate: bool = False) -> None:
    if recreate and os.path.lexists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def resolve_path(path: str | Path, strict=False) -> Path:
    return Path(path).resolve(strict=strict)


def short_relpath(path: str | Path, *, start: str | Path | None = None) -> Path:
    try:
        if start is None:
            rel = os.path.relpath(path)
        else:
            rel = os.path.relpath(path, start=start)
    except ValueError:
        return Path(path)
    return Path(rel)


def shorter_relpath(path: str | Path, *, start: str | Path | None = None) -> Path:
    rel = short_relpath(path, start=start)
    if len(str(rel)) < len(str(path)):
        return rel
    else:
        return Path(path)


def alt_filepath(filepath: str | Path, *, suffix: str = "+") -> Path:
    path = resolve_path(filepath)
    return path if not path.exists() else alt_filepath(path.with_stem(path.stem + suffix), suffix=suffix)


def build_filepath(dirpath: str | Path, filename: str, fileext: str, *, exist_ok: bool = True, suffix: str = "+", strict_dirpath: bool = True) -> Path:
    filepath = resolve_path(resolve_path(dirpath, strict=strict_dirpath) / (filename + os.extsep + fileext), strict=False)
    return filepath if exist_ok else alt_filepath(filepath, suffix=suffix)


def open_filepath_write(dirpath: str | Path, filename: str, fileext: str, *, exist_ok: bool = True, suffix: str = "+", binary: bool = True, **kwargs) -> IO:
    filepath = build_filepath(dirpath, filename, fileext)
    if exist_ok:
        return open(filepath, "wb" if binary else "w", **kwargs)
    else:
        while True:
            try:
                return open(filepath, "xb" if binary else "x", **kwargs)
            except FileExistsError:
                filepath = alt_filepath(filepath, suffix=suffix)


def glob_recursively(dirpath: str | Path, fileext: str) -> list[Path]:
    globs = resolve_path(dirpath, strict=True).rglob("*" + os.extsep + glob.escape(fileext), case_sensitive=False)
    return [f for f in (resolve_path(g, strict=True) for g in globs) if f.is_file()]
