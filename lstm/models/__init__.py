import torch

from .lstm import BiLSTM_max


def get_model(args):
    # creating the model
    if args.verbose:
        print('[+] creating model', args.model)

    model = BiLSTM_max(args)
    # convert model to GPU if possible
    if torch.cuda.is_available():
        if args.verbose:
            print('[+] converting model', args.model, 'to cuda')
            print(model)
        model.to(args.device)
    return model
