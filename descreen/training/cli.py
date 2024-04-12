import torch
import torch.cuda
import torch.backends.mps
from typing import Any
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from .proc import train
from ..networks.model.abs import DescreenModelType
from ..utilities.args import eqsign_kvpairs
from ..utilities.filesys import mkdirp, open_filepath_write


def main() -> int:
    exit_code = 0
    model_aliases = list(DescreenModelType.aliases.keys())
    parser = ArgumentParser(prog="python3 -m descreen.training", allow_abbrev=False, formatter_class=ArgumentDefaultsHelpFormatter, description="")
    parser.add_argument("train", metavar="TRAIN_PATH", help="describe directory")
    parser.add_argument("valid", metavar="VALID_PATH")
    parser.add_argument("test", metavar="TEST_PATH")
    parser.add_argument("-n", "--model-name", dest="name", metavar="MODEL", type=str, choices=model_aliases, required=True, help="")
    parser.add_argument("-p", "--model-params", dest="params", metavar="KWARGS", type=eqsign_kvpairs, required=True, help="")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


    dest: Path = args.dirc
    name: str = args.name
    kwargs: dict[str, Any] = args.params

    mkdirp()

    model = DescreenModelType.by_alias(name)(**kwargs)
    trained_model = train(model, args.train, args.valid, device=device)
    trained_model.to(torch.device("cpu"))
    with open_filepath_write(".", "model_weights", "ddbin", exist_ok=False) as fp:
        trained_model.serialize(fp)
    return exit_code
