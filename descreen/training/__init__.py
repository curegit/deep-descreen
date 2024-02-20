import torch
from torch import nn

from torch import Tensor

from torch.utils.data import DataLoader

from .data import HalftonePairDataset

def train(model, train_data_dir, valid_data_dir, device=None):
    model.to(device)
    model.train()
    batch_size = 8

    patch_size = 256
    loss_fn = nn.MSELoss()
    #loss_fn = nn.L1Loss()
    #optimizer = torch.optim.RAdam(model.parameters(), lr=0.001)
    optimizer = torch.optim.LBFGS(model.parameters(), lr=0.1)

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
                #valid_step()
                #test_step()
                continue
            #test_step()
            break


    def train_step(x, y ):
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

    train_loop(16)


        #if i % 100 == 0:
            #loss, current = loss.item(), (i + 1) * len(X)

            #print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")



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
