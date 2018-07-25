"""
File: custom_data_runner.py
Description:
  Implementation of data runner from
  https://indico.io/blog/tensorflow-data-input-part2-extensions/
"""
import sys
sys.path.insert(0, '/usr/local/lib/python2.7/dist-packages')

import threading

import tensorflow as tf

from data.fast_data_handler import FastDataHandler

USE_VEL = True
if USE_VEL:
  INPUT_SIZE = 14
else:
  INPUT_SIZE = 12

class CustomDataRunner():
  """Class to manage threads which fill a queue of data"""
  def __init__(self, filepath, batch_size, chunksize, max_perception_radius=30.0):
    with tf.device("/cpu:0"):
      self.batch_size = batch_size
      self.data_handler = FastDataHandler(filepath, batch_size, chunksize, laser_subsampling=True,
                                          max_perception_radius=max_perception_radius,
                                          use_odom_vel=USE_VEL)
      self.data_x = tf.placeholder(dtype=tf.float32, shape=[None, INPUT_SIZE])
      self.data_y = tf.placeholder(dtype=tf.float32, shape=[None, 2])

      # The actual queue of data. The queue contains a vector for
      # the mnist features, and a scalar label.
      self.queue = tf.RandomShuffleQueue(shapes=[[INPUT_SIZE], [2]],
                      dtypes=[tf.float32, tf.float32],
                      capacity=2000,
                      min_after_dequeue=1000)

      # The symbolic operation to add data to the queue
      # we could do some pre-processing here or do it in numpy. In this example
      # we do the scaling in numpy
      self.enqueue_op = self.queue.enqueue_many([self.data_x, self.data_y])
      self.threads_stop = False
      self.threads = list()
      self.sess = None

  def close(self):
    self.threads_stop = True
    self.data_handler.close()
    if self.sess:
      self.sess.run(self.queue.close(cancel_pending_enqueues=True))
    for thr in self.threads:
      thr.join()

  def get_inputs(self):
    """
    Return's tensors containing a batch of laser data and labels
    """
    data_batch, cmd_batch = self.queue.dequeue_many(self.batch_size)
    return data_batch, cmd_batch

  def thread_main(self, sess, coord):
    """
    Function run on alternate thread. Basically, keep adding data to the queue.
    """
    while not (coord.should_stop() or self.threads_stop):
      data_x, data_y =self.data_handler.next_batch()
      try:
        sess.run(self.enqueue_op, feed_dict={self.data_x:data_x, self.data_y:data_y})
      except tf.errors.CancelledError as e:
        # This happens if we stop processing and the enque operation is pending
        pass

  def start_threads(self, sess, coord, n_threads=1):
    """ Start background threads to feed queue """
    self.sess = sess
    threads = []
    for n in range(n_threads):
      t = threading.Thread(target=self.thread_main, args=(sess,coord))
      t.daemon = True
      t.start()
      self.threads.append(t)
