
import tensorflow as tf
from tensorflow.python.platform import gfile

import os
import argparse

def parse_args():
  """Parse input arguments."""
  parser = argparse.ArgumentParser(description='Save model to generate graph')
  parser.add_argument('graph_file', help='Path to the graph definition (.pb)')
  parser.add_argument('storage_path', help='Path to write the output')

  args = parser.parse_args()

  return args
def main():

  args = parse_args()

  with gfile.FastGFile(args.graph_file,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

  print('Loaded protobuf file: {}'.format(args.graph_file))

  with tf.Session() as sess:

    filename = os.path.join(args.storage_path, 'final')
    writer = tf.train.SummaryWriter(filename, sess.graph)

if __name__ == "__main__":
  main()
