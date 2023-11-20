import sys
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from descreening.training import train_loop, test_loop
from descreening.pitch.dataset import PitchImageArrayDataset
from descreening.pitch.model import PitchModel


from descreening.utilities import alt_filepath, build_filepath


def train(model, training_data, test_data, epochs, batch_size, device="cpu"):
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)

        # i = rgb_image_to_wide_gamut_uint16_array()
        # k = model.predict(i)
        # print()

    print("Done!")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = PitchModel().to(device)

    a = sys.argv[1]
    real_data = sys.argv[2]

    dataset = PitchImageArrayDataset(a, device=device)
    training_data, test_data = random_split(dataset, [0.8, 0.2])

    train(model, training_data, test_data, epochs=100, batch_size=32, device=device)

    path = alt_filepath(build_filepath(".", "pitch_model_weights", "pth"))
    torch.save(model.state_dict(), path)
    print(f"Saved: {path}")
