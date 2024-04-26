from ...utilities.filesys import self_relpath as rel
from .abs import DescreenModel


files = {
    "basic": rel("./unet/model.ddbin"),
}

names = list(files.keys())

default_name = names[0]

def pull(name: str):
    f = files.get(name)
    if f is None:
        raise ValueError()
    return DescreenModel.deserialize(f)


from .basic import *
from .unet import *
