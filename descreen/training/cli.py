from typing import Any
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from .proc import train
from ..networks.model.abs import DescreenModelType
from ..utilities.args import natural, directory, file, upper, eqsign_kvpairs, backend_device, backend_devices
from ..utilities.filesys import mkdirp, open_filepath_write, shorter_relpath


def main() -> int:
    exit_code = 0
    model_aliases = list(DescreenModelType.aliases.keys())
    exe = "python3 -m descreen.training"
    parser = ArgumentParser(prog=exe, allow_abbrev=False, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("train", metavar="TRAIN_PATH", help="describe dataset directory (train set)")
    parser.add_argument("valid", metavar="VALID_PATH")
    parser.add_argument("test", metavar="TEST_PATH")
    parser.add_argument("-d", "--dest", metavar="DIR", type=directory(exist=False), default=shorter_relpath("."), help="write output to target directory DIR")
    parser.add_argument("-m", "--model", dest="name", metavar="NAME", type=str, choices=model_aliases, required=True, help="")
    parser.add_argument("-p", "--params", dest="params", metavar="KWARGS", type=eqsign_kvpairs, required=True, help="")
    parser.add_argument("-e", "--max-epoch", dest="epoch", metavar="N", type=natural, help="aa")
    parser.add_argument("-c", "--profile", metavar="ICC", type=file(exist=True), help="aa")
    parser.add_argument("-z", "--device", type=upper, default="CPU", choices=backend_devices)
    args = parser.parse_args()

    dest: Path = args.dest
    name: str = args.name
    kwargs: dict[str, Any] = args.params
    device = backend_device(args.device)

    mkdirp(dest)
    model = DescreenModelType.by_alias(name)(**kwargs)
    trained_model, exit_code = train(model, args.train, args.valid, args.test, dest, max_epoch=args.epoch, profile=args.profile, device=device)
    trained_model.to(backend_device("CPU"))
    with open_filepath_write(dest, f"{name}-final", "ddbin", exist_ok=False) as fp:
        trained_model.serialize(fp)
    return exit_code
