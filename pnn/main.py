import torch
from torch import nn
from .training import train_loop, test_loop
from .dataset import CustomImageArrayDataset, CustomImageTensorDataset



def cli():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


    from .models import SillyModel
    model = SillyModel().to(device)

    patch_size = 256

    p = model.reduced_padding(patch_size)






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


    from .models import SillyModel
    model = SillyModel()
    model.load_state_dict(torch.load(sys.argv[1]))


    model.to(device)
    model.eval()
    print(model)

    patch_size = 256
    img = read_image(sys.argv[2])

    h, w = img.shape[1:3]
    img = img.reshape((1, 3, h, w))
    res = np.zeros((3, h, w), dtype="float32")
    p = model.required_padding(patch_size)
    img = np.pad(img, ((0, 0), (0, 0), (p, p), (p, p)), mode="symmetric")
    for (j, i), (k, l) in model.patch_slices(h, w, patch_size):
        x = img[:, :, j, i]
        t = torch.from_numpy((x / (2 ** 16 - 1)).astype("float32"))
        t = t.to(device)
        y = model(t)
        yy = y.detach().cpu().numpy()
        res[:, k, l] = yy[0]
    save_image(res, sys.argv[3])
