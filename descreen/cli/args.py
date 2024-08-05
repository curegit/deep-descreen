from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from descreen.networks.model import names, default_name
from descreen.cli.types import filelike, file, upper, backend_device, backend_devices

def parse():
    parser = ArgumentParser(prog="descreen", allow_abbrev=False, formatter_class=ArgumentDefaultsHelpFormatter, description="")
    parser.add_argument("image", metavar="IN_FILE", type=filelike(exist=True), help="describe directory")
    parser.add_argument("output", metavar="OUT_FILE", type=filelike(exist=False), nargs="?", default=..., help="describe input image files (pass '-' to specify stdin)")
    dest_group = parser.add_mutually_exclusive_group()
    dest_group.add_argument("-m", "--model", metavar="NAME", choices=names, default=default_name, help=f"send output to standard output {names}")
    dest_group.add_argument("-d", "--ddbin", metavar="FILE", type=file(exist=True), help="save output images in DIR directory")
    dest_group.add_argument("-x", "--onnx", metavar="FILE", type=file(exist=True), help="save output images in DIR directory")
    parser.add_argument("-q", "--quantize", "--depth", type=int, default=8, choices=[8, 16], help="color depth of output PNG")
    parser.add_argument("-z", "--device", type=upper, default="CPU", choices=backend_devices)
    args = parser.parse_args()
    return args
