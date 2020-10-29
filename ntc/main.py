from tqdm import trange

from models import util
import models

def main():
    # Handle parameters
    args = util.get_args()

    # Select gpu
    util.select_gpu(args)

    # Load data
    dataloader, dataloader_full, args = util.get_dataloader(args)

    # Create model
    model = models.get_model(args)

    # Print args
    util.print_args(args)

    # Create logger
    logger = util.Logger(args)

    # Begin training
    iterator = trange(args.num_epoch)
    try:
        for epoch in iterator:
            description = model.train(dataloader, dataloader_full, logger, epoch)
            iterator.set_description(description)
    except KeyboardInterrupt:
        pass

    # Printing result
    logger.display()

    # Saving result
    logger.save()

    # Saving model
    model.save(logger)

if __name__ == '__main__':
    main()
