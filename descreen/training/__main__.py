import torch
import torch.cuda
import torch.backends.mps
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from .proc import train

from ..networks.models import DescreenModelType


def main():
    parser = ArgumentParser(
        prog="python3 -m descreen.training",
        allow_abbrev=False,
        formatter_class=ArgumentDefaultsHelpFormatter,
        # description="Check if Unicode text files are Unicode-normalized",
        # prefix_chars="-",
    )
    parser.add_argument("train", metavar="PATH", help="describe directory")
    parser.add_argument("valid", metavar="PATH")
    parser.add_argument("test", metavar="PATH")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    print(device)

    model = DescreenModelType.by_alias(name)(**kwargs)
    trained_model = train(model, args.train, args.valid, device=device)
    trained_model.to(torch.device("cpu"))
    trained_model.serialize("model_weights.ddbin")


if __name__ == "__main__":
    main()
