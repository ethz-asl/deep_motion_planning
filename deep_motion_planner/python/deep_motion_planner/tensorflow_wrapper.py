
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

        found_keep_prob = False
        for o in self.sess.graph.get_operations():
            if 'keep_prob_placeholder' in o.name:
                found_keep_prob = True
                break

        if found_keep_prob:
            self.feed_dict = {'keep_prob_placeholder:0': 1.0}
        else:
            self.feed_dict = dict()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Make sure to close the session and clena up all the used resources
        self.sess.close()

    def process_attention(self, alpha):

        a_sensor = np.reshape(alpha, [alpha.shape[1]/64, 64]).transpose()

        return a_sensor

    def process_activation(self, activation):

        activation = activation.squeeze().transpose()

        return activation

    def inference(self, data):
        """
        Take the given data and perform the model inference on it
        """
        self.feed_dict['data_input:0'] = [data]

        mode = 0
        if mode == 0:
            # Return Attention matrix
            prediction, alpha = self.sess.run(['model_inference:0', 
                'Softmax_2:0'], feed_dict=self.feed_dict)

            cnn_data = self.process_attention(alpha)
        
        elif mode == 1:
            # Return activation matrix
            prediction, act = self.sess.run(['model_inference:0', 
                'AvgPool2D_2/AvgPool:0'], feed_dict=self.feed_dict)

            cnn_data = self.process_activation(act)

        elif mode == 2:
            # Only process the command
            prediction = self.sess.run(['model_inference:0'], feed_dict=self.feed_dict)[0]

            cnn_data = np.zeros([64,45])

        else:
            print('Unsupported mode: {}'.format(mode))
        

        return (prediction[0,0], prediction[0,1], cnn_data)

