import os
import pandas as pd
import numpy as np

class DataHandler():
    """Class to load data from HDF5 storages in a random and chunckwise manner"""
    def __init__(self, filepath, chunksize=1000):
        self.filepath = filepath
        self.chunksize = chunksize

        if not os.path.exists(filepath):
            raise IOError('File does not exists: {}'.format(filepath))

        # Get the number of rows without loading any data into memory
        with pd.HDFStore(filepath) as store:
            self.nrows = store.get_storer('data').nrows

        # Make sure, we can return a chunk with the correct size
        if chunksize > self.nrows:
            raise ValueError('Chunksize is to large for the dataset: {} chunks > {} rows'.format(chunksize, self.nrows))

        # Initialize the batches
        self.batches = self.__next_permutation__()

    def __next_permutation__(self):
        """
        Generator for the permutations. 

        @return Indices of the elements in the batch
        @rtype Generator
        """
        current_index = 0
        permutation = np.random.permutation(self.nrows)
        while True 
            yield permutation[current_index:current_index+self.chunksize]
            current_index += self.chunksize

            # We iterated over the entire dataset, so resample and reset
            if (current_index + self.chunksize) > self.nrows:
                permutation = np.random.permutation(self.nrows)
                current_index = 0

    def next_batch(self):
        """
        Load the next random batch from the loaded data file

        @return (input data, ground truth commands)
        @rtype Tuple
        """
        # Load the data for the next batch
        ind = next(self.batches)
        df = pd.read_hdf(self.filepath, 'data', where='index=ind')

        # Separate the data into the returned frames
        data_columns = [column for column in df.columns 
                if column.split('_')[0] in ['laser','target'] and not column.split('_')[1] == 'id']

        data = df[data_columns]
        cmd = df[['linear_x','angular_z']]
        return (data,cmd)
            


        
