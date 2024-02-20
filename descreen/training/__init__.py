import torch
from torch import nn

from torch import Tensor




def train(model, x_loader, y_loader):

    def train_loop(iters: int):
        i = 0
        epoch = 1
        while True:
            for k, (x, y) in enumerate(dataloader):
                X = X.to(device)
                y = y.to(device)
                train_step(x, y)
                i += 1
                if i >= iters:
                    break
            else:
                valid_step()
                test_step()
                continue
            test_step()
            break


    def train_step(x, y ):
        size = len(dataloader.dataset)
        # Set the model to training mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        model.train()


            """
            # Compute prediction and loss
            pred = model(X)
            #print(pred.shape)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            """
            loss = None
            def clos():
                global loss
                pred = model(X)
                loss = loss_fn(pred, y)
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                return loss

            optimizer.step(clos)


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
