from . import util

import torch

from .nocnn import NoCNN
from .ntc import NTC
from .cp import CP

models = {
    'nocnn': NoCNN,
    'ntc': NTC,
    'cp': CP,
}

def get_model(args):
    if args.verbose:
        print('[+] creating model', args.model)
    # Creating model
    if args.model in models.keys():
        model = models[args.model](args)
    else:
        raise NotImplementedError
     # convert model to GPU if possible
    if torch.cuda.is_available():
        if args.verbose:
            print('[+] converting model', args.model, 'to cuda')
            print(model)
        model = model.cuda()
    return model


