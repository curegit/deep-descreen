from descreen.assets import grab_file
from descreen.networks.model.abs import DescreenModel


files = {
    "basic": "basic.ddbin",
}

names = list(files.keys())

default_name = names[0]


def pull(name: str):
    f = files.get(name)
    if f is None:
        raise ValueError()
    with grab_file("models", f) as p:
        return DescreenModel.deserialize(p)


# TODO: dynamic?
from descreen.networks.model.basic import *
from descreen.networks.model.unet import *
