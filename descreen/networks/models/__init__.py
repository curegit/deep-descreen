from .basic import TopLevelModel
from ...utilities.filesys import self_relpath as rp


files = {
    "basic": (TopLevelModel, rp("")),
}

names = list(files.keys)

def get(name: str):
    cls, file = files[name]
    m = cls.load(file)
    return m


