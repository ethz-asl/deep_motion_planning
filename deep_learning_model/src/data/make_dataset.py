# -*- coding: utf-8 -*-
import os
import logging
import argparse
from dotenv import find_dotenv, load_dotenv

import pandas as pd

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Fuse csv files and prepare data')
    parser.add_argument('input_path', help='Folder of the raw data')

    def check_extension(extensions, filename):
        ext = os.path.splitext(filename)[1]
        if ext not in extensions:
            parser.error("Unsupported file extension: Use {}".format(extensions))
        return filename

    parser.add_argument('output_file', help='Filename of the processed data (stored in \
            ./data/processed)', type=lambda
            s:check_extension([".h5"],s))
    args = parser.parse_args()

    return args

def main(project_dir):
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    args = parse_args()

    path = os.path.join(project_dir, args.input_path)
    all_files = [os.path.join(path,f) for f in os.listdir(path) 
                         if os.path.isfile(os.path.join(path, f)) and f.split('.')[-1] == 'csv']

    all_files.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))

    target_file = os.path.join(project_dir, 'data', 'processed', args.output_file)

    if os.path.exists(target_file):
        logger.error('Target file already exists: {}'.format(target_file))
        exit()

    store = pd.HDFStore(target_file)
    num_elements = 0

    for i,f in enumerate(all_files):
        print('({}/{}) {}'.format(i, len(all_files), f), end='\r')
        current = pd.read_csv(f)
        num_elements += current.shape[0]
                            
        store.append('data', current)
            
    print('')
    store.close()
    logger.info('We combined {} lines'.format(num_elements))
    logger.info('File saved: {}'.format(target_file))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main(project_dir)

