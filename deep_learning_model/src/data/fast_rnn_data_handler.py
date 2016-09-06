
import threading
import time
import os
import pandas as pd
import numpy as np

class FastRNNDataHandler():
    """Class to load data from HDF5 storages in sequential order for RNNs"""
    def __init__(self, filepath, batchsize = 8, unrollings=1):
        self.filepath = filepath
        self.batchsize = batchsize
        self.unrollings = unrollings

        if not os.path.exists(filepath):
            raise IOError('File does not exists: {}'.format(filepath))

        with pd.HDFStore(filepath, mode='r') as store:
            # Get the number of rows without loading any data into memory
            self.nrows = store.get_storer('data').nrows

            target_ids = store.select('data', iterator=True, chunksize=10000, columns=['target_id'])

            # Get unique elements and the first index it occurs. Keep the order of occurance
            ids = np.array([])
            id_start_index = np.array([])
            for chunk in target_ids:
                current_targets = chunk['target_id'].values
                _, idx = np.unique(current_targets, return_index=True)
                ids = np.concatenate((ids, current_targets[np.sort(idx)] ))
                id_start_index = np.concatenate((id_start_index, chunk.index.values[np.sort(idx)]))

            _, idx = np.unique(ids, return_index=True)
            ids = ids[np.sort(idx)].astype(int).tolist()
            id_start_index = id_start_index[np.sort(idx)].astype(int).tolist()
            self.target_ids = [x for x in zip(ids, id_start_index)]

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

    def __generate_next_batch__(self):
        """
        Generator for the single batches
        """
        data_columns = None
        cmd_columns = None

        while True:

            # Load the first set of trajectories
            with pd.HDFStore(self.filepath, mode='r') as store:
                chunk = store.select('data', start=self.target_ids[0][1],
                        stop=self.target_ids[self.batchsize][1])

            for i in range(len(self.target_ids) // self.batchsize - 1):

                # Load the next set of trajectories in a separate thread
                self.chunk_thread = threading.Thread(target=self.next_chunk, args=(i+1,))
                self.chunk_thread.start()

                # On the first call, get the column indecies for the input data and the commands
                if not data_columns:
                    laser_columns = list()
                    target_columns = list()
                    cmd_columns = list()
                    for j,column in enumerate(chunk.columns):
                        if column.split('_')[0] == 'laser':
                            laser_columns.append(j)
                        if column.split('_')[0] == 'target'\
                            and not column.split('_')[1] == 'id':
                            target_columns.append(j)
                        if column in ['linear_x','angular_z']:
                            cmd_columns.append(j)

                    # Only use the center n_scans elements as input
                    n_scans = 1080
                    drop_n_elements = (len(laser_columns) - n_scans) // 2

                    if drop_n_elements < 0:
                        raise ValueError('Number of scans is to small: {} < {}'
                                .format(len(laser_columns), n_scans))
                    elif drop_n_elements > 0:
                        laser_columns = laser_columns[drop_n_elements:-drop_n_elements]

                    if len(laser_columns) == n_scans+1:
                        laser_columns = laser_columns[0:-1]
                    
                    data_columns = laser_columns + target_columns

                trajectories = list()

                # Determine the lenght of the shortest trajctory that is covered
                # by unrollings without remainder. As the trajectories are ordered by their 
                # length, we can take the first one in the batch
                start_index = self.target_ids[i*self.batchsize][1]
                stop_index = self.target_ids[i*self.batchsize+1][1]
                shortest_trajectory = ((stop_index - start_index) // self.unrollings 
                        * self.unrollings)

                batch_start_index = start_index
                for j in range(self.batchsize):
                    
                    # Get a view onto a single trajectory used in the batches
                    # Also adjust the lenght to the shortest trajectory by 
                    # cutting elements from the beginning or the end

                    start_index = self.target_ids[i*self.batchsize+j][1] - batch_start_index
                    stop_index = self.target_ids[i*self.batchsize+(j+1)][1] - batch_start_index
                    # Cut elements at the start
                    trajectory_length = stop_index - start_index

                    if np.random.binomial(1, 0.5) > 0:
                        start_index = start_index + (trajectory_length - shortest_trajectory)
                    else:
                        # Cut elements at the end
                        stop_index = stop_index - (trajectory_length - shortest_trajectory)

                    trajectories.append(chunk.iloc[start_index:stop_index])

                # Return the batches from the current data chunk that is in memory
                for j in range(shortest_trajectory // self.unrollings):
                    data = np.zeros([self.batchsize, self.unrollings, len(data_columns)])
                    cmd = np.zeros([self.batchsize, self.unrollings, len(cmd_columns)])
                    for k in range(self.batchsize):
                        start_index = j*self.unrollings
                        end_index   = (j+1)*self.unrollings
                        data[k,0:self.unrollings,:] = trajectories[k].iloc[
                                start_index:end_index, data_columns].values
                        cmd[k,0:self.unrollings,:] = trajectories[k].iloc[
                                start_index:end_index, cmd_columns].values

                    yield (data, cmd)

                # Copy the preloaded trajectories for the next run
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
            self.buffer = store.select('data', start=self.target_ids[i*self.batchsize][1],
                    stop=self.target_ids[(i+1)*self.batchsize][1])

            self.new_buffer_data = True

    def next_batch(self):
        """
        Load the next random batch from the loaded data file

        @return (input data, ground truth commands)
        @rtype Tuple
        """
        # Load the data for the next batch
        return next(self.batches)

        
