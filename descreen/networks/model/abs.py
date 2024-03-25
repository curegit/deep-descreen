import copy
import json
import struct
import contextlib
import numpy as np
import safetensors.torch
from io import IOBase, BytesIO
from pathlib import Path
from abc import ABCMeta, abstractmethod
from numpy import ndarray
from .. import AbsModule
from ...utilities import range_chunks
from ...utilities.filesys import resolve_path


class DescreenModelType(ABCMeta):

    aliases: dict[str, type] = {}

    def __new__(meta, name, bases, attributes, **kwargs):
        cls = super().__new__(meta, name, bases, attributes, **kwargs)
        try:
            alias = cls.alias()
        except Exception:
            return cls
        if alias in DescreenModelType.aliases:
            raise RuntimeError()
        DescreenModelType.aliases[alias] = cls
        return cls

    @staticmethod
    def by_alias(alias: str):
        cls = DescreenModelType.aliases[alias]
        if issubclass(cls, DescreenModel):
            return cls
        else:
            raise RuntimeError()


class DescreenModel(AbsModule, metaclass=DescreenModelType):

    params_json: dict[int, str] = {}

    def __new__(cls, **kwargs):
        obj = super().__new__(cls)
        params_json = json.dumps(kwargs, skipkeys=False, ensure_ascii=True, allow_nan=False)
        DescreenModel.params_json[id(obj)] = params_json
        return obj

    def __copy__(self):
        cp = copy.copy(super())
        DescreenModel.params_json[id(cp)] = DescreenModel.params_json[id(self)]
        return cp

    def __deepcopy__(self, memo):
        cp = copy.deepcopy(super(), memo)
        DescreenModel.params_json[id(cp)] = DescreenModel.params_json[id(self)]
        return cp

    @classmethod
    @abstractmethod
    def alias(cls) -> str:
        raise NotImplementedError()

    @property
    @abstractmethod
    def multiple_of(self) -> int:
        return 1

    def patch(self, x: ndarray, input_patch_size: int, *, pad_mode: str = "symmetric"):
        if input_patch_size % self.multiple_of != 0:
            raise ValueError()
        out_patch_size = self.output_size(input_patch_size)
        p = self.required_padding(out_patch_size)
        height, width = x.shape[-2:]
        qh = input_patch_size - self.patch_slices_remainder(height + 2 * p, input_patch_size, p)
        qw = input_patch_size - self.patch_slices_remainder(width + 2 * p, input_patch_size, p)
        pad_width: tuple[tuple[int, int], ...] = (*([(0, 0)] * (x.ndim - 2)), (p, qh + p), (p, qw + p))
        y = np.pad(x, pad_width, mode=pad_mode)
        h_crop = slice(p, height + p)
        w_crop = slice(p, width + p)
        return y, self.patch_slices(height + qh, width + qw, out_patch_size, p), (h_crop, w_crop)

    def patch_slices(self, height: int, width: int, output_patch_size: int, padding: int):
        for h_start, h_stop in range_chunks(height, output_patch_size):
            for w_start, w_stop in range_chunks(width, output_patch_size):
                assert h_stop - h_start == w_stop - w_start == output_patch_size
                h_slice = slice(h_start - padding + padding, h_stop + padding + padding)
                w_slice = slice(w_start - padding + padding, w_stop + padding + padding)
                h_dest_slice = slice(h_start + padding, h_stop + padding)
                w_dest_slice = slice(w_start + padding, w_stop + padding)
                yield (h_slice, w_slice), (h_dest_slice, w_dest_slice)

    def patch_slices_remainder(self, length: int, input_patch_size: int, padding: int):
        cur = 0
        while length - cur >= input_patch_size:
            cur += input_patch_size - padding * 2
        return length - cur

    def load_weight(self, buffer: bytes) -> None:
        self.load_state_dict(safetensors.torch.load(buffer))

    @classmethod
    def load(cls, byteslike: bytes | IOBase):
        match byteslike:
            case bytes() as bin:
                ctx = fp = BytesIO(bin)
            case IOBase() as fp:
                ctx = contextlib.nullcontext()
            case _:
                raise TypeError()
        with ctx:
            (l,) = struct.unpack(jsp := "!I", fp.read(struct.calcsize(jsp)))
            js = fp.read(l).decode()
            kwargs = json.loads(js)
            model = cls(**kwargs)
            model.load_weight(fp.read())
        return model

    @staticmethod
    def deserialize(filelike: str | Path | bytes | IOBase):
        fp: IOBase
        match filelike:
            case str() | Path() as path:
                ctx = fp = open(resolve_path(path, strict=True), "rb")
            case bytes() as bin:
                ctx = fp = BytesIO(bin)
            case IOBase() as fp:
                ctx = contextlib.nullcontext()
            case _:
                raise TypeError()
        with ctx:
            (l,) = struct.unpack(ap := "!H", fp.read(struct.calcsize(ap)))
            alias = fp.read(l).decode()
            cls = DescreenModelType.by_alias(alias)
            model = cls.load(fp)
        return model

    def serialize_weight(self) -> bytes:
        return safetensors.torch.save(self.state_dict(), metadata=None)

    def serialize(self, filelike: str | Path | IOBase) -> None:
        match filelike:
            case str() | Path() as path:
                ctx = fp = open(resolve_path(path), "wb")
            case IOBase() as fp:
                ctx = contextlib.nullcontext()
            case _:
                raise TypeError()
        with ctx:
            ab = self.alias().encode()
            fp.write(struct.pack("!H", len(ab)))
            fp.write(ab)
            jsb = DescreenModel.params_json[id(self)].encode()
            fp.write(struct.pack("!I", len(jsb)))
            fp.write(jsb)
            fp.write(self.serialize_weight())
