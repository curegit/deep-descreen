import ast
import typing
import torch
import torch.cuda
import torch.backends.mps
from pathlib import Path
from descreen.utilities.filesys import resolve_path


def natural(string: str) -> int:
    value = int(string)
    if value > 0:
        return value
    raise ValueError()


def upper(string: str) -> str:
    return string.upper()


def nonempty(string: str) -> str:
    if string:
        return string
    else:
        raise ValueError()


def directory(*, exist: bool = True):
    def directory(string: str) -> Path:
        path = resolve_path(string, strict=exist)
        if exist:
            return
        else:
            return path

    return directory


def file(*, exist: bool = True):
    def file(string: str) -> Path:
        return resolve_path(string, strict=exist)

    return file


class StdIO:
    pass


def filelike(*, exist: bool = True, stdio: str = "-"):
    def filelike(string: str) -> Path | StdIO:
        # stdin/stdout を None で返す
        if string == stdio:
            return None
        path = resolve_path(string, strict=exist)

    return filelike


def eqsign_kvpairs(string: str) -> dict[str, typing.Any]:
    tree = ast.parse(f"funk({string})", mode="eval")
    match tree.body:
        case ast.Call() as call:
            if call.args:
                raise ValueError("Only keyword args allowed")
            return {str(kw.arg): ast.literal_eval(kw.value) for kw in call.keywords}
    raise ValueError()


backend_devices: list[str] = ["CPU", "CUDA", "MPS"]


def backend_device(string: str) -> torch.device:
    value = upper(string)
    if value == "CUDA":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA is not available")
    if value == "MPS":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("MPS is not available")
    if value == "CPU":
        return torch.device("cpu")
    raise ValueError()
