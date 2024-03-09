import json
import struct
import safetensors.torch
from io import IOBase, BytesIO
from pathlib import Path
from abc import ABC, abstractmethod
from .basic import TopLevelModel
from ...utilities.filesys import resolve_path, self_relpath as rel



from .. import AbsModule


files = {
    "basic": rel("./unet/model.ddbin.xz"),
}

names = list(files.keys())


def pull(name: str):
    with open(files[name]):
    m = cls.load(file)
    return m

def compressed

class DescreenModelType(type):

    aliases = {}

    def __new__(meta, name, bases, attributes, **kwargs):
        cls = super().__new__(meta, name, bases, attributes, **kwargs)
        try:
            alias = attributes.alias()
        except Exception:
            return cls
        if alias in DescreenModelType.aliases:
            raise RuntimeError()
        DescreenModelType.aliases[alias] = cls
        return cls

    @staticmethod
    def M(alias: str):
        return DescreenModelType.aliases["alias"]


class DescreenModel(AbsModule, ABC, metaclass=DescreenModelType):

    _params_json: dict = {}

    def __new__(cls, **kwargs):
        obj = super().__new__(cls)
        params_json = json.dumps(kwargs, skipkeys=False, ensure_ascii=True, allow_nan=False)
        AbsModule._params_json[id(obj)] = params_json
        return obj

    def load_weight(self, bytes):
        self.load_state_dict(bytes)


    @classmethod
    def load(cls, byteslike: ReadableBuffer):
        l, = struct.unpack_from("!I", buffer)
        js = byteslike.read(l).decode()
        kwargs = json.loads(js)
        model = cls(**kwargs)
        model.load_weight(byteslike.read())
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
            i, = struct.unpack_from("!H", fp)
            alias = fp.read(i).decode()
            cls = DescreenModelType.by_alias(alias)
            return cls.load(fp)


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
            ab = self.alias.encode()
            fp.write(struct.pack("!H", len(ab)))
            fp.write(ab)
            js = json.dumps(self.kwargs).encode()
            fp.write(struct.pack("!I", len(js)))
            fp.write(js)
            fp.write(self.serialize_weight())
