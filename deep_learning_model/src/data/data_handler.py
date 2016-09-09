import os
import pandas as pd
import numpy as np

class DataHandler():
    """Class to load data from HDF5 storages in a random and chunckwise manner"""
    def __init__(self, filepath, chunksize=1000, shuffle = True):
        self.filepath = filepath
        self.chunksize = chunksize
        self.shuffle = shuffle
        self.perception_radius = 10.0

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
        if self.shuffle:
            permutation = np.random.permutation(self.nrows)
        else:
            permutation = np.arange(self.nrows)
        while True:
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

        laser_columns = list()
        goal_columns = list()
        cmd_columns = list()
        for j,column in enumerate(df.columns):
            if column.split('_')[0] in ['laser']: 
                laser_columns.append(j)
            if column.split('_')[0] in ['target'] and not column.split('_')[1] == 'id':
                goal_columns.append(j)
            if column in ['linear_x','angular_z']:
                cmd_columns.append(j)

        #Only use the center n_scans elements as input
        n_scans = 1080
        drop_n_elements = (len(laser_columns) - n_scans) // 2

	if drop_n_elements < 0:
		raise ValueError('Number of scans is to small: {} < {}'
			.format(len(laser_columns), n_scans))
	elif drop_n_elements > 0:
		laser_columns = laser_columns[drop_n_elements:-drop_n_elements]

	if len(laser_columns) == n_scans+1:
		laser_columns = laser_columns[0:-1]
        
        laser = np.minimum(df.iloc[:,laser_columns].values, self.perception_radius)
        goal = df.iloc[:,goal_columns].values
        angle = np.arctan2(goal[:,1],goal[:,0]).reshape([self.chunksize, 1])
        norm = np.minimum(np.linalg.norm(goal[:,0:2], ord=2, axis=1).reshape([self.chunksize, 1]), self.perception_radius)
        data = np.concatenate((laser, angle, norm, goal[:,2].reshape([self.chunksize, 1])), axis=1)

        return (data.copy(), df.iloc[:, cmd_columns].copy(deep=True).values)

        
