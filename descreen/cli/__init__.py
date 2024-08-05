import sys
import numpy as np
import torch
from io import BytesIO
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from descreen.image import load_image, save_image
from descreen.image.magick import magick_wide_png, magick_srgb_png
from descreen.networks.model import DescreenModel, pull, names, default_name
from descreen.cli.args import parse
from descreen.cli.types import upper, nonempty, file, filelike, backend_device, backend_devices


def main() -> int:
    exit_code = 0

    args = parse()

    device = backend_device(args.device)

    if args.ddbin is not None:
        model = DescreenModel.deserialize(args.ddbin)
    elif args.onnx is not None:
        # TODO
        pass
    else:
        model = pull(args.model)
    model.to(device)
    model.eval()
    print(model)

    # img = read_uint16_image(sys.argv[3])

    if args.image is None:
        in_bin = sys.stdin.buffer.read()
    else:
        with open(args.image, "rb") as fp:
            in_bin = fp.read()
    img = load_image(magick_wide_png(in_bin, relative=True), assert16=True)
    padded, patches, crop = model.patch(img, 512)
    dest = np.ones_like(padded, dtype=np.float32)
    for (j, i), (k, l) in patches:
        x = padded[:, j, i].astype(np.float32)
        t = torch.from_numpy(x).reshape((1, *x.shape)).to(device)
        z = model(t)
        y = z.detach().cpu().numpy()[0]
        print(y.shape)
        dest[:, k, l] = y
    result = dest[:, crop[0], crop[1]]

    buf = BytesIO()
    save_image(result, buf, prefer16=True, compress=False)
    r = magick_srgb_png(buf.getvalue(), relative=True, prefer48=(args.quantize == 16), assume_wide=True, radical=True)
    if args.output is None:
        pass
    if args.output is Ellipsis:
        if args.image is None:
            i = "a"
        else:
            i = args.image.stem + "-descreen"
        output = (Path(".") / i).with_suffix(".png")
    else:
        output = args.output
    with open(output, "wb") as fp:
        fp.write(r)
