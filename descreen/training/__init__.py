import torch
from torch import nn

from torch import Tensor
from torch.optim.swa_utils import AveragedModel
from torch.optim.swa_utils import get_ema_multi_avg_fn
from torch.utils.data import DataLoader

from .data import HalftonePairDataset
from .loss import total_variation

from ..networks.models import DescreenModel
from ..networks.models.unet import UNetLikeModel
from ..networks.models.basic import TopLevelModel

model = UNetLikeModel()
ac = torch.nn.functional.leaky_relu
# model = TopLevelModel(128, ac, 8)


def train(model, train_data_dir, valid_data_dir, device=None) -> DescreenModel:
    model.to(device)
    model.train()
    amodel = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))
    batch_size = 8

    patch_size = 364
    loss_fn = nn.MSELoss()
    # loss_fn = nn.L1Loss()
    optimizer = torch.optim.RAdam(model.parameters(), lr=0.001)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.002)
    # optimizer = torch.optim.LBFGS(model.parameters(), lr=0.1)

    p = model.reduced_padding(patch_size)
    print(p)

    print("OutputSize:", model.output_size(patch_size))

    pro = "JapanColor2011Coated.icc"
    training_data = HalftonePairDataset(train_data_dir, pro, patch_size, p)
    valid_data = HalftonePairDataset(valid_data_dir, pro, patch_size, p)

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8, prefetch_factor=4, persistent_workers=True)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8, prefetch_factor=4, persistent_workers=True)

    def train_loop(iters: int):
        i = 0
        epoch = 1
        while True:
            for k, (x, y) in enumerate(train_dataloader):
                i += 1
                print(f"iter {i}")
                X = x.to(device)
                Y = y.to(device)
                train_step(X, Y)

                if i >= iters:
                    break
            else:
                # valid_step()
                # test_step()
                continue
            # test_step()
            break

    def train_step(x, y):
        if False:
            loss = None

            def clos():
                global loss
                pred = model(x)
                loss = loss_fn(pred, y)
                print(f"loss: {loss}")
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                return loss

            optimizer.step(clos)
        else:
            pred = model(x)
            loss = loss_fn(pred, y) + (0.5 * total_variation(pred)).mean()
            optimizer.zero_grad()
            loss.backward()
            print(f"loss: {loss}")
            optimizer.step()
            amodel.update_parameters(model)

    try:
        train_loop(3000)
    except KeyboardInterrupt:
        pass
    rm = amodel.module
    print(rm)
    assert isinstance(rm, DescreenModel)
    return rm

    # if i % 100 == 0:
    # loss, current = loss.item(), (i + 1) * len(X)

    # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def valid_step(dataloader, model, loss_fn, device):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def test_step():
    pass
