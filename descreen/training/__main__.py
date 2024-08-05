import sys
from .cli import main

import torch.multiprocessing

import multiprocessing


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    print(torch.multiprocessing.get_start_method())
    #sys.exit()
    sys.exit(main())
