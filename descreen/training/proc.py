import signal
import torch
import torch.optim
import torch.optim.swa_utils
from pathlib import Path
from torch.utils.data import DataLoader
from . import patch_size, batch_size, num_images
from .loss import descreen_loss
from .data import HalftonePairDataset, enumerate_loader
from ..networks.model import DescreenModel

from descreen.utilities.filesys import open_filepath_write

def train[
    T: DescreenModel
](
    model: T,
    train_data_dir: str | Path,
    valid_data_dir: str | Path,
    test_data_dir: str | Path,
    output_dir: str | Path,
    *,
    max_epoch: int | None = None,
    profile: str | Path | None,
    device: torch.device,
) -> tuple[T, int]:
    exit_code = 0



    model.to(device)
    model.train()
    ema_model = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.996))
    assert isinstance(model, DescreenModel) and isinstance(ema_model.module, type(model))

    #optimizer = torch.optim.RAdam(model.parameters(), lr=0.001)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    input_size = model.input_size(patch_size)
    padding = model.reduced_padding(input_size)
    assert model.output_size(input_size) == patch_size

    print("InputSize:", input_size)
    print("OutputSize:", patch_size)

    training_data = HalftonePairDataset(train_data_dir, profile, input_size, padding, augment=True, debug=True, debug_dir=output_dir, extend=batch_size * 30).as_tensor()
    valid_data = HalftonePairDataset(valid_data_dir, profile, input_size, padding).as_tensor()

    def train_loop(max_samples: int, *, graceful: bool = True):
        interrupted = False
        default_sigint = None

        def interrupt(signum, frame):
            nonlocal interrupted
            first_interruption = not interrupted
            interrupted = True

        if graceful:
            default_sigint = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, signal.SIG_IGN)

        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=30, prefetch_factor=3, persistent_workers=False, pin_memory=True, multiprocessing_context="spawn")
        valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=30, prefetch_factor=3, persistent_workers=False, pin_memory=True, multiprocessing_context="spawn")
        signal.signal(signal.SIGINT, interrupt)
        last_epoch = 0
        for (epoch, iters, samples), (x, y) in enumerate_loader(train_dataloader, device=device):
            if interrupted:
                if callable(default_sigint):
                    signal.signal(signal.SIGINT, default_sigint)
                break
            if samples > max_samples or (max_epoch is not None and epoch > max_epoch):
                valid_step(valid_dataloader, iters)
                test_step()
                break
            print(f"epoch {epoch}")
            print(f"iters {iters}")
            print(f"samples {samples}")
            train_step(x, y, iters)
            if iters and iters % 100 == 0:
                valid_step(valid_dataloader, iters)
            if last_epoch != epoch:
                last_epoch = epoch
                test_step()

    def train_step(x, y, i):
        model.train()
        pred_mid, pred_full = model.forward_t(x)
        loss = descreen_loss(pred_mid, y, tv=0.03) + 2.4 * descreen_loss(pred_full, y, tv=0.01)
        optimizer.zero_grad()
        loss.backward()
        print(f"loss: {loss}")
        optimizer.step()
        ema_model.update_parameters(model)

    def valid_step(dataloader, i):
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
                test_loss += torch.nn.MSELoss()(pred, y).item()

        test_loss /= num_batches
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        with open_filepath_write(output_dir, f"{i}-ema", "ddbin", exist_ok=False) as fp:
            ema_model.module.serialize(fp)
        with open_filepath_write(output_dir, f"{i}", "ddbin", exist_ok=False) as fp:
            model.serialize(fp)

    def test_step():
        pass

    train_loop(num_images)

    result_model = ema_model.module
    assert isinstance(result_model, DescreenModel)
    return result_model, exit_code
