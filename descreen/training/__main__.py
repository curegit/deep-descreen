


from . import train


import sys
import torch
from io import BytesIO
from ..models.unet import UNetLikeModel
from ..image import load_image, save_image, magick_wide_png, magick_srgb_png
import numpy as np
import sys

def main():

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model = UNetLikeModel()


    train(model, "data", "data", device=device)

    torch.save(model.state_dict(), 'model_weights.pth')



if __name__ == "__main__":
    main()
