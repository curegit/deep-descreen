import torch
from .training import train
from .dataset import CustomImageArrayDataset, CustomImageTensorDataset


def cmain():
    out_size = 512
    p = model.required_padding(out_size)



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

