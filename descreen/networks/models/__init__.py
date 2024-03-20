import json
import struct
import safetensors.torch
import numpy as np
from io import IOBase, BytesIO
from pathlib import Path
from abc import ABCMeta, abstractmethod
from numpy import ndarray
from .. import AbsModule
from ...utilities import range_chunks
from ...utilities.filesys import resolve_path, self_relpath as rel


files = {
    "basic": rel("./unet/model.ddbin"),
}

names = list(files.keys())


def pull(name: str):
    return DescreenModel.deserialize(files[name])


class DescreenModelType(ABCMeta):

    aliases = {}

    def __new__(meta, name, bases, attributes, **kwargs):
        cls = super().__new__(meta, name, bases, attributes, **kwargs)
        print("new")
        try:
            # alias = attributes["alias"]()
            alias = cls.alias()
        except Exception:
            raise
            # return cls
        if alias in DescreenModelType.aliases:
            raise RuntimeError()
        DescreenModelType.aliases[alias] = cls
        return cls

    @staticmethod
    def by_alias(alias: str):
        print(DescreenModelType.aliases)
        return DescreenModelType.aliases[alias]


class DescreenModel(AbsModule, metaclass=DescreenModelType):

    _params_json: dict = {}

    def __new__(cls, **kwargs):
        obj = super().__new__(cls)
        params_json = json.dumps(kwargs, skipkeys=False, ensure_ascii=True, allow_nan=False)
        DescreenModel._params_json[id(obj)] = params_json
        return obj

    @classmethod
    @abstractmethod
    def alias(cls) -> str:
        print(cls.__name__)
        return cls.__name__

    @property
    def multiple_of(self) -> int:
        return 1

    def patch(self, x: ndarray, input_patch_size: int, *, pad_mode: str = "symmetric", **kwargs):
        if input_patch_size % self.multiple_of != 0:
            raise RuntimeError()
        out_patch_size = self.output_size(input_patch_size)
        p = self.required_padding(out_patch_size)
        assert out_patch_size > 0
        print(p)
        height, width = x.shape[-2:]

        # self.patch_slices(height, width, out_patch_size, p)

        qh = input_patch_size - self.patch_slices_remainder(height + 2 * p, input_patch_size, p)
        qw = input_patch_size - self.patch_slices_remainder(width + 2 * p, input_patch_size, p)
        y = np.pad(x, (*([(0, 0)] * (x.ndim - 2)), (p, qh + p), (p, qw + p)), mode=pad_mode, **kwargs)
        h_crop = slice(p, height + p)
        w_crop = slice(p, width + p)
        return y, self.patch_slices(height + qh, width + qw, out_patch_size, p), (h_crop, w_crop)

    def patch_slices(self, height: int, width: int, output_patch_size: int, padding: int):
        for h_start, h_stop in range_chunks(height, output_patch_size):
            for w_start, w_stop in range_chunks(width, output_patch_size):
                h_pad = padding  # self.required_padding(h_stop - h_start)
                w_pad = padding  # self.required_padding(w_stop - w_start)
                print(h_stop - h_start, w_stop - w_start, output_patch_size)
                assert h_stop - h_start == w_stop - w_start == output_patch_size
                h_slice = slice(h_start - h_pad + padding, h_stop + h_pad + padding)
                w_slice = slice(w_start - w_pad + padding, w_stop + w_pad + padding)
                h_dest_slice = slice(h_start + padding, h_stop + padding)
                w_dest_slice = slice(w_start + padding, w_stop + padding)
                yield (h_slice, w_slice), (h_dest_slice, w_dest_slice)

    def patch_slices_remainder(self, length: int, input_patch_size: int, padding: int):
        cur = 0
        while length - cur >= input_patch_size:
            cur += input_patch_size - padding * 2
        return length - cur

    def load_weight(self, buffer: bytes):
        self.load_state_dict(safetensors.torch.load(buffer))

    @classmethod
    def load(cls, byteslike: bytes | IOBase):
        match byteslike:
            case bytes() as bin:
                buf = BytesIO(bin)
            case IOBase() as buf:
                pass
            case _:
                raise TypeError()
        (l,) = struct.unpack(f := "!I", buf.read(struct.calcsize(f)))
        js = buf.read(l).decode()
        kwargs = json.loads(js)
        model = cls(**kwargs)
        model.load_weight(buf.read())
        return model

    @staticmethod
    def deserialize(filelike: str | Path | bytes | IOBase):
        match filelike:
            case str() | Path() as path:
                fp = open(resolve_path(path, strict=True), "rb")
            case bytes() as bin:
                fp = BytesIO(bin)
            case IOBase() as fp:
                pass
            case _:
                raise TypeError()
        with fp:
            (i,) = struct.unpack(a := "!H", fp.read(struct.calcsize(a)))
            alias = fp.read(i).decode()
            cls = DescreenModelType.by_alias(alias)
            model: DescreenModel = cls.load(fp)
        return model

    def serialize_weight(self) -> bytes:
        return safetensors.torch.save(self.state_dict(), metadata=None)

    def serialize(self, filelike: str | Path | IOBase):
        match filelike:
            case str() | Path() as path:
                fp = open(resolve_path(path), "wb")
            case IOBase() as fp:
                pass
            case _:
                raise TypeError()
        with fp:
            ab = self.alias().encode()
            fp.write(struct.pack("!H", len(ab)))
            fp.write(ab)
            js = DescreenModel._params_json[id(self)].encode()
            fp.write(struct.pack("!I", len(js)))
            fp.write(js)
            fp.write(self.serialize_weight())


from .unet import UNetLikeModel

UNetLikeModel()
