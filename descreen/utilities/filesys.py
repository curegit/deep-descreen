import os
import os.path
import glob
import shutil
import inspect
from pathlib import Path
from typing import IO


def mkdirp(path: str | Path, recreate: bool = False) -> None:
    if recreate and os.path.lexists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def resolve_path(path: str | Path, strict=False) -> Path:
    return Path(path).resolve(strict=strict)


def self_relpath(relpath: str) -> Path:
    filename = inspect.stack()[1].filename
    dirpath = Path(filename).resolve().parent
    return (dirpath / relpath).resolve()


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
    pattern = os.path.join(glob.escape(str(resolve_path(dirpath, strict=True))), "**", "*" + os.extsep + glob.escape(fileext))
    globs = (resolve_path(f) for f in glob.glob(pattern, recursive=True))
    return [f for f in globs if f.is_file()]


def relaxed_glob_recursively(dirpath: str | Path, fileext: str) -> list[Path]:
    lower, upper = fileext.lower(), fileext.upper()
    ls = glob_recursively(dirpath, lower)
    if lower == upper:
        return ls
    ls_upper = glob_recursively(dirpath, upper)
    case_insensitive = len(ls) == len(ls_upper) > 0 and any(f.samefile(ls_upper[0]) for f in ls)
    if case_insensitive:
        return ls
    ls += ls_upper
    cap = fileext.capitalize()
    if cap == lower or cap == upper:
        return ls
    return ls + glob_recursively(dirpath, cap)
