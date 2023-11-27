import torch
from torch import nn
from .training import train_loop, test_loop
from .dataset import CustomImageArrayDataset, CustomImageTensorDataset

from .models.unet import UNetLikeModel
from .image import read_uint16_image, save_wide_gamut_uint16_array_as_srgb
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

    patch_size = 512

    p = model.reduced_padding(patch_size)
    print(p)

    print("OutputSize:", model.output_size(patch_size))






    training_data = CustomImageTensorDataset(CustomImageArrayDataset("data", patch_size), p, device)
    test_data = CustomImageTensorDataset(CustomImageArrayDataset("data_test", patch_size), p, device)



    train(model, training_data, test_data, 2200, 11)

    torch.save(model.state_dict(), 'model_weights.pth')


def train(model, training_data, test_data, epochs, batch_size, device="cpu"):
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")




def convert():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


    #from .models import UNetLikeModel
    model = UNetLikeModel()
    model.load_state_dict(torch.load(sys.argv[2]))


    model.to(device)
    model.eval()
    print(model)

    patch_size = model.output_size(512)
    img = read_uint16_image(sys.argv[3])

    #img = img[:,:-3,:]
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
        t = torch.from_numpy((x / (2 ** 16 - 1)).astype("float32"))
        t = t.to(device)
        y = model(t)
        yy = y.detach().cpu().numpy()
        print(y.shape)
        res[:, k, l] = yy[0]
    res = res[:, :h, :w]
    save_wide_gamut_uint16_array_as_srgb(res, sys.argv[4])
