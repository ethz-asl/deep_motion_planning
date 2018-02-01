import sys
sys.path.insert(0, '/usr/local/lib/python2.7/dist-packages')

import threading
import time
import os
import pandas as pd
import numpy as np

class FastDataHandler():
  """Class to load data from HDF5 storages in a random and chunckwise manner"""
  def __init__(self, filepath, batchsize = 16, chunksize=None, maximum_perception_radius=10, mean_filter_size=5):
    self.filepath = filepath
    self.chunksize = chunksize
    self.batchsize = batchsize
    self.perception_radius = maximum_perception_radius
    self.mean_filter_size = mean_filter_size

    # Check if the parameters are valid

    # Chunksize needs to be big enough (if specified, otherwise whole data will be used)
    if chunksize and not chunksize % batchsize == 0:
      raise IOError('chunksize must be divisible by batchsize')

    # Path has to exist
    if not os.path.exists(filepath):
      raise IOError('File does not exists: {}'.format(filepath))

    # Get the number of rows without loading any data into memory
    with pd.HDFStore(filepath, mode='r') as store:
      self.nrows = store.get_storer('data').nrows

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
    odom_vel_columns = None

    while True:
      current_index = 0

      if self.use_chunks:
        with pd.HDFStore(self.filepath, mode='r') as store:
          chunk = store.select('data',
              start=0, stop=self.chunksize)

      for i in range(self.nrows // self.chunksize - 1):

        # Load the next chunk of data from the HDF5 container
        if self.use_chunks:
          self.chunk_thread = threading.Thread(target=self.next_chunk, args=(i+1,))
          self.chunk_thread.start()
        else:
          with pd.HDFStore(self.filepath, mode='r') as store:
            chunk = store.select('data')

        # Apply rolling mean on the velocity values

        # Velocity commands
        chunk['filtered_linear_command'] = pd.Series.rolling(chunk['linear_x_command'],
            window=self.mean_filter_size, center=True).mean().fillna(chunk['linear_x_command'])
        chunk['filtered_angular_command'] = pd.Series.rolling(chunk['angular_z_command'],
            window=self.mean_filter_size, center=True).mean().fillna(chunk['angular_z_command'])

        # Velocity measurements (odometry)
        chunk['filtered_linear_odom'] = pd.Series.rolling(chunk['linear_x_odom'],
            window=self.mean_filter_size, center=True).mean().fillna(chunk['linear_x_odom'])
        chunk['filtered_angular_odom'] = pd.Series.rolling(chunk['angular_z_odom'],
            window=self.mean_filter_size, center=True).mean().fillna(chunk['angular_z_odom'])

        # Shuffle the data
        chunk = chunk.reindex(np.random.permutation(chunk.index))

        # On the first call, get the column indices for the input data and the commands
        if not laser_columns or not goal_columns or not cmd_columns or not odom_vel_columns:
          laser_columns = list()
          goal_columns = list()
          cmd_columns = list()
          odom_vel_columns = list()

          # Extract each column and check its name
          for j,column in enumerate(chunk.columns):
            if column.split('_')[0] in ['laser']:
              laser_columns.append(j)
            if column.split('_')[0] in ['target'] and not column.split('_')[1] == 'id':
              goal_columns.append(j)
            if column in ['filtered_linear_command','filtered_angular_command']:
              cmd_columns.append(j)
            if column in ['filtered_linear_odom','filtered_angular_odom']:
              odom_vel_columns.append(j)

          # Only use the center n_scans elements as input
          n_scans = 1080

          # Compute number of elements to be dropped per side
          drop_n_elements = (len(laser_columns) - n_scans) // 2

          # Remove elements to be dropped on each side
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

          # Laser samples of size n_scans
          laser = np.minimum(chunk.iloc[j*self.batchsize:(j+1)*self.batchsize,laser_columns].values,
              self.perception_radius)

          # Goal data: distance, angle, heading (in robot frame)
          goal =  chunk.iloc[j*self.batchsize:(j+1)*self.batchsize,goal_columns].values
          angle = np.arctan2(goal[:,1],goal[:,0]).reshape([self.batchsize, 1])
          norm = np.minimum(np.linalg.norm(goal[:,0:2], ord=2,
            axis=1).reshape([self.batchsize, 1]), self.perception_radius)


          # Velocity measurements (current velocity of robot)
          odom_vel = chunk.iloc[j*self.batchsize:(j+1)*self.batchsize, odom_vel_columns].values

          # Concatenate data: laser, goal angle, goal distance, goal heading
          data = np.concatenate((laser, angle, norm, goal[:,2].reshape([self.batchsize,1]), odom_vel), axis=1)

          yield (data.copy(), chunk.iloc[j*self.batchsize:(j+1)*self.batchsize, cmd_columns].values)
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


