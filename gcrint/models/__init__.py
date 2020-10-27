import torch

from .gcrint import GCRINT


def get_model(args):
    # creating the model
    if args.verbose:
        print('[+] creating model', args.model)
    if args.model == 'gcrint':
        model = GCRINT(args)
    else:
        raise NotImplementedError
    # convert model to GPU if possible
    if torch.cuda.is_available():
        if args.verbose:
            print('[+] converting model', args.model, 'to cuda')
            print(model)
        model.to(args.device)
    return model
