import os
import pandas as pd
import numpy as np
import support as sup

class DataHandler():
  """Class to load data from HDF5 storages in a random and chunkwise manner"""
  def __init__(self, filepath, chunksize=1000, shuffle = True, laser_subsampling = False, num_dist_values = 36):
    self.filepath = filepath
    self.chunksize = chunksize
    self.shuffle = shuffle
    self.perception_radius = 10.0
    self.mean_filter_size = 5
    self.use_odom_vel = False
    self.laser_subsampling = True
    self.num_dist_values = num_dist_values

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

    # Add new entry to data with filtered velocities
    df['filtered_linear_command'] = pd.Series.rolling(df['linear_x_command'],
        window=self.mean_filter_size, center=True).mean().fillna(df['linear_x_command'])
    df['filtered_angular_command'] = pd.Series.rolling(df['angular_z_command'],
        window=self.mean_filter_size, center=True).mean().fillna(df['angular_z_command'])

    if self.use_odom_vel:
      df['filtered_linear_odom'] = pd.Series.rolling(df['linear_x_odom'],
        window=self.mean_filter_size, center=True).mean().fillna(df['linear_x_odom'])
      df['filtered_angular_odom'] = pd.Series.rolling(df['angular_z_odom'],
        window=self.mean_filter_size, center=True).mean().fillna(df['angular_z_odom'])

    laser_columns = list()
    goal_columns = list()
    cmd_columns = list()
    for j,column in enumerate(df.columns):
      if column.split('_')[0] in ['laser']:
        laser_columns.append(j)
      if column.split('_')[0] in ['target'] and not column.split('_')[1] == 'id':
        goal_columns.append(j)
      if column in ['filtered_linear_command','filtered_angular_command']:
        cmd_columns.append(j)
      if column in ['filtered_linear_odom','filtered_angular_odom']:
            odom_vel_columns.append(j)

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
    if self.laser_subsampling:
      laser = sup.subsample_laser(laser, self.num_dist_values)
    goal = df.iloc[:,goal_columns].values
    angle = np.arctan2(goal[:,1],goal[:,0]).reshape([self.chunksize, 1])
    norm = np.minimum(np.linalg.norm(goal[:,0:2], ord=2, axis=1).reshape([self.chunksize, 1]), self.perception_radius)

    # Velocity measurements (current velocity of robot)
    if self.use_odom_vel:
      odom_vel = df.iloc[j*self.batchsize:(j+1)*self.batchsize, odom_vel_columns].values

    # Concatenate data: laser, goal angle, goal distance, goal heading
    if self.use_odom_vel:
      data = np.concatenate((laser, angle, norm, goal[:,2].reshape([self.chunksize, 1]), odom_vel), axis=1)
    else:
      data = np.concatenate((laser, angle, norm, goal[:,2].reshape([self.chunksize, 1])), axis=1)

    return (data.copy(), df.iloc[:, cmd_columns].copy(deep=True).values)


