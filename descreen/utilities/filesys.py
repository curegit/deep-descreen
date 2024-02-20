import os
import os.path
import glob
import shutil
import inspect
from pathlib import Path


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
