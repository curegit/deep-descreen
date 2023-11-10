import torch
from torch import nn
from .training import train_loop, test_loop
from .dataset import CustomImageArrayDataset, CustomImageTensorDataset


def main():
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






    training_data = CustomImageTensorDataset(CustomImageArrayDataset("data", patch_size), p)
    test_data = CustomImageTensorDataset(CustomImageArrayDataset("data_test", patch_size), p)



    train(model, training_data, test_data, 200, 2)
    
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

    
