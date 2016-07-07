
import threading
import time
import os
import pandas as pd
import numpy as np

class FastDataHandler():
    """Class to load data from HDF5 storages in a random and chunckwise manner"""
    def __init__(self, filepath, batchsize = 16, chunksize=None):
        self.filepath = filepath
        self.chunksize = chunksize
        self.batchsize = batchsize

        # Check if the parameters are valid
        if chunksize and not chunksize % batchsize == 0:
            raise IOError('chunksize must be divisible by batchsize')

        if not os.path.exists(filepath):
            raise IOError('File does not exists: {}'.format(filepath))

        # Get the number of rows without loading any data into memory
        store = pd.HDFStore(filepath, mode='r')
#         with pd.HDFStore(filepath, mode='r') as store:
        self.nrows = store.get_storer('data').nrows
        store.close()

        # No chunksize specified, load the entire dataset
        if not self.chunksize:
            self.chunksize = self.nrows
            self.use_chunks = False
        else:
            self.use_chunks = True
            self.chunk_thread = None
            self.interrupt_thread = False

        self.buffer = None
        self.new_buffer_data = False

        self.batches = self.__generate_next_batch__()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        self.interrupt_thread = True
        if self.chunk_thread and self.chunk_thread.is_alive():
            self.chunk_thread.join()

    def steps_per_epoch(self):
        """
        Get the number of steps to process the entire dataset once
        """
        return self.nrows // self.batchsize

    def __generate_next_batch__(self):
        """
        Generator for the single batches
        """
        laser_columns = None
        goal_columns = None
        cmd_columns = None

        while True:
            current_index = 0

            if self.use_chunks:
#                 with pd.HDFStore(self.filepath, mode='r') as store:
                store = pd.HDFStore(self.filepath, mode='r')
                chunk = store.select('data',
                                     start=0, stop=self.chunksize)
                store.close()

            for i in range(self.nrows // self.chunksize - 1):

                # Load the next chunk of data from the HDF5 container
                if self.use_chunks:
                    self.chunk_thread = threading.Thread(target=self.next_chunk, args=(i+1,))
                    self.chunk_thread.start()
                else:
                    with pd.HDFStore(self.filepath, mode='r') as store:
                        chunk = store.select('data')

                # Shuffle the data
                chunk = chunk.reindex(np.random.permutation(chunk.index))

                # On the first call, get the column indecies for the input data and the commands
                if not laser_columns:
                    laser_columns = list()
                    goal_columns = list()
                    cmd_columns = list()
                    for j,column in enumerate(chunk.columns):
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
                    
                    data_columns = laser_columns + goal_columns


                # Return the batches from the current data chunk that is in memory
                for j in range(chunk.shape[0] // self.batchsize):
                    
                    laser = chunk.iloc[j*self.batchsize:(j+1)*self.batchsize,laser_columns].values
                    goal =  chunk.iloc[j*self.batchsize:(j+1)*self.batchsize,goal_columns].values
                    angle = np.arctan2(goal[:,1],goal[:,0]).reshape([self.batchsize, 1]) / np.pi
                    norm = np.minimum(np.linalg.norm(goal[:,0:2], ord=2,
                        axis=1).reshape([self.batchsize, 1]), 10.0) / 10.0
                    data = np.concatenate((laser, angle, norm, goal[:,2].reshape([self.batchsize,
                        1])/np.pi), axis=1)

                    yield (data.copy(), 
                        chunk.iloc[j*self.batchsize:(j+1)*self.batchsize, cmd_columns].values
)
                    current_index += self.batchsize

                if self.use_chunks:
                    start_time = time.time()
                    self.chunk_thread.join()
                    got_data = False
                    while not got_data:
                        if self.new_buffer_data:
                            chunk = self.buffer.copy(deep=True)
                            self.new_buffer_data = False
                            got_data = True
                        else:
                            time.sleep(0.5)
                    print('Waited: {}'.format(time.time() - start_time))

                if self.interrupt_thread:
                    return

    def next_chunk(self, i):
        with pd.HDFStore(self.filepath, mode='r') as store:
            self.buffer = store.select('data',
                    start=i*self.chunksize, stop=(i+1)*self.chunksize)
            self.new_buffer_data = True

    def next_batch(self):
        """
        Load the next random batch from the loaded data file

        @return (input data, ground truth commands)
        @rtype Tuple
        """
        # Load the data for the next batch
        return next(self.batches)

        
