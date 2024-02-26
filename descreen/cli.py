import sys
import torch
from io import BytesIO
from .models.unet import UNetLikeModel
from .models.simple import TopLevelModel
from .image import load_image, save_image, magick_wide_png, magick_srgb_png

import numpy as np

from .training import model

def main():
    device = "cpu"

    from torch.optim.swa_utils import AveragedModel
    global model
    model1 = AveragedModel(model)
    model = model1
    model.load_state_dict(torch.load(sys.argv[1]))


    model.to(device)
    model.eval()
    print(model)
    model = model.module

    patch_size = model.output_size(512)
    #img = read_uint16_image(sys.argv[3])

    with open(sys.argv[2], "rb") as fp:
        img = load_image(magick_wide_png(fp.read(), relative=True), assert16=True)


    height, width = img.shape[1:3]
    # TODO: 4倍数にあわせる
    ppp_h = height % 512
    ppp_w = width % 512
    a_h = height + ppp_h
    a_w = width + ppp_w
    img = img.reshape((1, 3, height, width))
    res = np.zeros((3, a_h, a_w), dtype="float32")
    p = model.required_padding(patch_size)

    img = np.pad(img, ((0, 0), (0, 0), (p, p + ppp_h), (p, p + ppp_w)), mode="symmetric")
    for (j, i), (k, l) in model.patch_slices(a_h, a_w, patch_size):
        print(k)
        x = img[:, :, j, i]
        t = torch.from_numpy(x.astype("float32"))
        t = t.to(device)
        y = model(t)
        yy = y.detach().cpu().numpy()
        print(y.shape)
        res[:, k, l] = yy[0]
        #break
    buf = BytesIO()
    save_image(res, buf, prefer16=True)
    r = magick_srgb_png(buf.getvalue(), relative=True, prefer48=False)
    with open(sys.argv[3], "wb") as fp:
        fp.write(r)
    #save_wide_gamut_uint16_array_as_srgb(res, sys.argv[4])
