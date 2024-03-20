import torch
import torch.cuda
import torch.backends.mps
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from .proc import train
from ..networks.model.abs import DescreenModelType
from ..utilities.args import eqsign_kvpairs


def main() -> int:
    exit_code = 0
    model_aliases = list(DescreenModelType.aliases.keys())
    parser = ArgumentParser(
        prog="python3 -m descreen.training",
        allow_abbrev=False,
        formatter_class=ArgumentDefaultsHelpFormatter,
        description=""
    )
    parser.add_argument("train", metavar="TRAIN_PATH", help="describe directory")
    parser.add_argument("valid", metavar="VALID_PATH")
    parser.add_argument("test", metavar="TEST_PATH")
    parser.add_argument("-n", "--model-name", metavar="MODEL", type=str, choices=model_aliases, required=True, help="")
    parser.add_argument("-p", "--model-params", metavar="KWARGS", type=eqsign_kvpairs, required=True, help="")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    print(device)
    name = args.name
    kwargs = args.params

    model = DescreenModelType.by_alias(name)(**kwargs)
    trained_model = train(model, args.train, args.valid, device=device)
    trained_model.to(torch.device("cpu"))
    trained_model.serialize("model_weights.ddbin")
    return exit_code
