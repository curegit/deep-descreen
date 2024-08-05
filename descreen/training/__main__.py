import sys
import torch.multiprocessing
from descreen.training.cli import main

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    print(f"MultiProcessing Method: {torch.multiprocessing.get_start_method()}")
    sys.exit(main())
