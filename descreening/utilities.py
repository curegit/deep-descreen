import sys
import inspect
import os
import os.path
import pathlib
import glob
import numpy as np


def range_chunks(length, n):
    for i in range(0, length, n):
        if i + n < length:
            yield i, i + n
        else:
            yield i, length

def mkdirs(dirpath):
	os.makedirs(os.path.normpath(dirpath), exist_ok=True)


def save_array(filepath, array):
	np.save(filepath, array, allow_pickle=False)

def load_array(filepath):
	return np.load(filepath, allow_pickle=False)


def alt_filepath(filepath, suffix="+"):
    while os.path.lexists(filepath):
        root, ext = os.path.splitext(filepath)
        head, tail = os.path.split(root)
        filepath = os.path.join(head, tail + suffix) + ext
    return filepath

def build_filepath(dirpath, filename, fileext, exist_ok=True, suffix="+"):
    filepath = os.path.normpath(os.path.join(dirpath, filename) + os.extsep + fileext)
    return filepath if exist_ok else alt_filepath(filepath, suffix)

def glob_shallowly(dirpath, fileext):
    pattern = build_filepath(glob.escape(dirpath), "*", glob.escape(fileext))
    return [f for f in glob.glob(pattern) if os.path.isfile(f)]

def glob_recursively(dirpath, fileext):
    pattern = build_filepath(glob.escape(dirpath), os.path.join("**", "*"), glob.escape(fileext))
    return [f for f in glob.glob(pattern, recursive=True) if os.path.isfile(f)]

def relaxed_glob_recursively(dirpath, fileext):
    lower, upper = fileext.lower(), fileext.upper()
    ls = glob_recursively(dirpath, lower)
    if lower == upper:
        return ls
    ls_upper = glob_recursively(dirpath, upper)
    case_insensitive = len(ls) == len(ls_upper) > 0 and any(os.path.samefile(f, ls_upper[0]) for f in ls)
    if case_insensitive:
        return ls
    ls += ls_upper
    cap = fileext.capitalize()
    if cap == lower or cap == upper:
        return ls
    return ls + glob_recursively(dirpath, cap)

def file_rel_path(relpath):
	filename = inspect.stack()[1].filename
	dirpath = pathlib.Path(filename).resolve().parent
	return str(dirpath.joinpath(relpath).resolve())


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
