import argparse
import logging
import os

from src.data.data_handler import DataHandler

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Train our model on the given dataset')

    def check_extension(extensions, filename):
        ext = os.path.splitext(filename)[1]
        if ext not in extensions:
            parser.error("Unsupported file extension: Use {}".format(extensions))
        return filename

    parser.add_argument('datafile', help='Filename of the training data', type=lambda
            s:check_extension(['.h5'],s))
    parser.add_argument('-s', '--max_steps', help='Number of batches to run', default=1000000)
    parser.add_argument('-b', '--batch_size', help='Size of training batches', default=16)
    args = parser.parse_args()

    return args

def main():
    logger = logging.getLogger(__name__)
    logger.info('Train our model on the given dataset:')

    args = parse_args()
    logger.info(args.datafile)

    # Open data handler
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    data = DataHandler(os.path.join(project_dir,args.datafile), args.batch_size)
    logger.info('Dataset loaded')

    logger.info('Start training')
    for i in range(args.max_steps):
        (X,Y) = data.next_batch()

        # TODO Add training
    

if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
