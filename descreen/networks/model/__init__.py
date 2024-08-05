from descreen.utilities.filesys import self_relpath as rel
from descreen.networks.model.abs import DescreenModel


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

# TODO: dynamic?
from descreen.networks.model.basic import *
from descreen.networks.model.basic import *
