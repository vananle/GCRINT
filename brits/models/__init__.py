import torch

from .brits import Brits


def get_model(args):
    # creating the model
    if args.verbose:
        print('[+] creating model', args.model)
    if args.model == 'brits':
        model = Brits(args.num_hidden, 1, 0, args.seq_len - 2, args.dim, args.device)
    else:
        raise NotImplementedError
    # convert model to GPU if possible
    if torch.cuda.is_available():
        if args.verbose:
            print('[+] converting model', args.model, 'to cuda')
            print(model)
        model = model.to(args.device)
    return model
