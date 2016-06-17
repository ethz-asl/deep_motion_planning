
from tensorflow.python.platform import gfile
import tensorflow as tf
import os
import sys
    
class TensorflowWrapper():
    """
    The class is used to load a pretrained tensorflow model and 
    use it on live sensor data
    """
    def __init__(self, storage_path, protobuf_file='graph.pb', use_checkpoints=False):
        
        with gfile.FastGFile(os.path.join(storage_path, protobuf_file),'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')

        print('Loaded protobuf file: {}'.format(os.path.join(storage_path, protobuf_file)))

        self.sess = tf.Session()

        if use_checkpoints:
            ckpt = tf.train.get_checkpoint_state(storage_path)
            if ckpt and ckpt.model_checkpoint_path:
                print('Found checkpoints: \n{}'.format(ckpt))
                print('Load: {}'.format(ckpt.model_checkpoint_path))
                self.sess.run(['save/restore_all'], {'save/Const:0': os.path.join(storage_path,
                    ckpt.model_checkpoint_path)})
            else:
                print('No checkpoint files in folder: {}'.format(storage_path))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.sess.close()

    def inference(self, data):
        feed_dict = {'data_input:0': [data]}

        prediction = self.sess.run(['prediction:0'], feed_dict=feed_dict)[0]

        return (prediction[0,0], prediction[0,1])

