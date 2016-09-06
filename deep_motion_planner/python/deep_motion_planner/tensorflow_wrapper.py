
from tensorflow.python.platform import gfile
import tensorflow as tf
import os
import sys
import numpy as np
    
class TensorflowWrapper():
    """
    The class is used to load a pretrained tensorflow model and 
    use it on live sensor data
    """
    def __init__(self, storage_path, protobuf_file='graph.pb', use_checkpoints=False):
        """Initialize a new TensorflowWrapper object
        
        @param storage_path: The path to the protobuf_file and the snapshots
        @type  :  string
        
        @param protobuf_file: The protobuf file to load
        @type  :  string
        
        @param use_checkpoints: If the given protobuf_file does not contain the trained weights, you
                                    can use checkpoint files to initialize the weights
        @type  :  bool
        
        """ 
        # Load the graph definition from a binary protobuf file
        with gfile.FastGFile(os.path.join(storage_path, protobuf_file),'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')

        print('Loaded protobuf file: {}'.format(os.path.join(storage_path, protobuf_file)))

        self.sess = tf.Session()

        # If the protobuf file does not contain the trained weights, you can load them from
        # a checkpoint file in the storage_path folder
        if use_checkpoints:
            # Get all checkpoints
            ckpt = tf.train.get_checkpoint_state(storage_path)
            if ckpt and ckpt.model_checkpoint_path:
                # Open the model with the highest step count (the last snapshot)
                print('Found checkpoints: \n{}'.format(ckpt))
                print('Load: {}'.format(ckpt.model_checkpoint_path))
                self.sess.run(['save/restore_all'], {'save/Const:0': os.path.join(storage_path,
                    ckpt.model_checkpoint_path)})
            else:
                print('No checkpoint files in folder: {}'.format(storage_path))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Make sure to close the session and clena up all the used resources
        self.sess.close()

    def inference(self, data):
        """
        Take the given data and perform the model inference on it
        """
        feed_dict = {'data_input:0': [data]}

        prediction = self.sess.run(['model_inference:0'], feed_dict=feed_dict)[0]

        return (prediction[0,0], prediction[0,1])

