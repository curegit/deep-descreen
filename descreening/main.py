import torch
from torch import nn
from .training import train_loop, test_loop
from .dataset import CustomImageArrayDataset_, CustomImageTensorDataset

from .models.unet import UNetLikeModel
from .image import save_image, read_uint16_image, save_wide_gamut_uint16_array_as_srgb
import sys
import numpy as np

def cli():
    if len(sys.argv) >= 2:
        if sys.argv[1] == "convert":
            convert()
            return

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )



    from .models.unet import UNetLikeModel
    model = UNetLikeModel().to(device)

    patch_size = 256

    p = model.reduced_padding(patch_size)
    print(p)

    print("OutputSize:", model.output_size(patch_size))






    training_data = CustomImageTensorDataset(CustomImageArrayDataset_("data_", patch_size), p, device)
    test_data = CustomImageTensorDataset(CustomImageArrayDataset_("data_test_", patch_size), p, device)



    train(model, training_data, test_data, 300, 16, device=device)

    torch.save(model.state_dict(), 'model_weights.pth')


def train(model, training_data, test_data, epochs, batch_size, device="cpu"):
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8, prefetch_factor=4, persistent_workers=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8, prefetch_factor=4, persistent_workers=True)

    loss_fn = nn.MSELoss()
    #loss_fn = nn.L1Loss()
    #optimizer = torch.optim.RAdam(model.parameters(), lr=0.001)
    optimizer = torch.optim.LBFGS(model.parameters(), lr=0.1)
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, device=device)
        test_loop(test_dataloader, model, loss_fn, device=device)
        torch.save(model.state_dict(), f'model_weights-{t}.pth')
    print("Done!")



from PIL import Image
from PIL.Image import Resampling
from numpy import rint, asarray, uint8, float32

def from_pil_image(img):
      return (asarray(img.convert("RGB"), dtype=uint8).transpose(2, 0, 1) / 255).astype(float32)



def convert():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    device = "cpu"


    #from .models import UNetLikeModel
    model = UNetLikeModel()
    model.load_state_dict(torch.load(sys.argv[2]))


    model.to(device)
    model.eval()
    print(model)

    patch_size = model.output_size(512)
    #img = read_uint16_image(sys.argv[3])
    im = Image.open(sys.argv[3])
    img = from_pil_image(im)

    img = img[:,:-1,:-1]
    h, w = img.shape[1:3]
    # TODO: 4倍数にあわせる
    ppp_h = h % 512
    ppp_w = w % 512
    a_h = h + ppp_h
    a_w = w + ppp_w
    img = img.reshape((1, 3, h, w))
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
    res = res[:, :h, :w]
    save_image(res, sys.argv[4])
    #save_wide_gamut_uint16_array_as_srgb(res, sys.argv[4])
