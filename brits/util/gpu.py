import torch


def select_gpu(args):
    if torch.cuda.is_available():
        if args.verbose:
            print('[+] selecting gpu: {}'.format(args.gpu))
        torch.cuda.set_device(args.gpu)
    else:
        if args.verbose:
            print('[+] selecting gpu: none selected')
