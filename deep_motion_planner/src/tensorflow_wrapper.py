
import tensorflow as tf
import os
import sys
    
project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
path = os.path.join(project_dir, 'deep_learning_model','src','model')

if path not in sys.path:
    sys.path.insert(0, path)

import model

class TensorflowWrapper():
    """
    The class is used to load a pretrained tensorflow model and 
    use it on live sensor data
    """
    def __init__(self):
        
        self.data_placeholder = tf.placeholder(tf.float32, shape=[None, 723])
        self.prediction = model.inference(self.data_placeholder)

        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

        saver = tf.train.Saver()
        model_path = os.path.join(path, os.pardir, os.pardir, 'models','2016-06-15_14-28_two_fully_connected')
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.sess, os.path.join(model_path, ckpt.model_checkpoint_path))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.sess.close()

    def inference(self, data):
        feed_dict = {self.data_placeholder: [data]}

        prediction = self.sess.run([self.prediction], feed_dict=feed_dict)[0]

        return (prediction[0,0], prediction[0,1])

