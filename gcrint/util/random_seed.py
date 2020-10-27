import random

import numpy as np
import torch


def set_random_seed(args):
    if args.verbose:
        print('[+] setting random, numpy, torch random seed: {}'.format(args.seed))
    # Python's
    random.seed(args.seed)
    # Numpy's
    np.random.seed(args.seed)
    # PyTorch's
    torch.manual_seed(args.seed)
