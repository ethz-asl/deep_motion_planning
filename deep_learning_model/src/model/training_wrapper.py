import _init_paths

import logging 
import os
import time
import getpass

import tensorflow as tf
import numpy as np

from data.custom_data_runner import CustomDataRunner
from data.data_handler import DataHandler
import conv_model as model

class TrainingWrapper():
    """Wrap the training"""
    def __init__(self, args):
        self.args = args
        self.coord = None
        self.runners = None
        self.sess = None
        self.custom_data_runner = None
        self.eval_n_elements = 60000
        self.eval_batch_size = 1024

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.custom_data_runner:
            self.custom_data_runner.close()
        if self.coord:
            self.coord.request_stop()
            self.coord.join(self.runners)
        if self.sess:
            self.sess.close()

    def placeholder_inputs(self, data_size, cmd_size, data_placeholder_name='data_input'):
        """Create placeholders for the tf graph"""
        data_placeholder = tf.placeholder(tf.float32, shape=[None, data_size], name=data_placeholder_name)
        cmd_placeholder = tf.placeholder(tf.float32, shape=[None, cmd_size])

        return data_placeholder, cmd_placeholder

    def check_extend(self, a, rows):
        """Check if the array a has the correct number of rows. In case it is to small pad it with zeros"""
        if a.shape[0] < rows:
            a = np.concatenate((a, np.zeros((rows-a.shape[0], a.shape[1]))), axis=0)
        
        return a

    def run(self):
        logger = logging.getLogger(__name__)

        # Folder where to store snapshots, meta data and the final model
        storage_path = os.path.join(self.args.train_dir, (time.strftime('%Y-%m-%d_%H-%M_') + model.NAME))

        logger.info('Build Tensorflow Graph')
        with tf.Graph().as_default():

            # Define the used machine learning model
            global_step, learning_rate = model.learning_rate(self.args.learning_rate)

            self.sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=8))

            logger.info('Create the data runner for the input data')
            self.custom_data_runner =  CustomDataRunner(self.args.datafile_train, self.args.batch_size, 2**18)
            data_batch, cmd_batch = self.custom_data_runner.get_inputs()

            logger.info('Add operations to the computation graph')
            keep_prob_placeholder = tf.placeholder(tf.float32, name='keep_prob_placeholder')
            prediction = model.inference(data_batch, keep_prob_placeholder, self.args.batch_size, output_name='prediction')

            loss, loss_split = model.loss(prediction, cmd_batch)

            train_op = model.training(loss, loss_split, learning_rate, global_step)

            eval_data_placeholder, eval_cmd_placeholder = self.placeholder_inputs(1083, 2, 'eval_data_input')
            eval_prediction = model.inference(eval_data_placeholder, keep_prob_placeholder,
                    self.eval_batch_size, training=False, reuse=True, output_name='eval_prediction')
            eval_predictions_placeholder = tf.placeholder(tf.float32, shape=[self.eval_n_elements,2])
            evaluation, evaluation_split = model.evaluation(eval_predictions_placeholder, eval_cmd_placeholder)

            # This model is saved with the trained weights and can direclty be executed
            exe_data_placeholder, exe_cmd_placeholder = self.placeholder_inputs(1083, 2)
            model_inference = model.inference(exe_data_placeholder, keep_prob_placeholder, 1,
                    training=False, reuse=True, output_name='model_inference')

            # Variables to use in the summary (shown in tensorboard)
            train_loss = tf.scalar_summary('loss', loss)
            train_loss_lin = tf.scalar_summary('loss_linear_x', loss_split[0])
            train_loss_ang = tf.scalar_summary('loss_angular_yaw', loss_split[1])
            train_learning_rate = tf.scalar_summary('learning_rate', learning_rate)

            eval_loss = tf.scalar_summary('loss', evaluation)
            eval_loss_lin = tf.scalar_summary('loss_linear_x', evaluation_split[0])
            eval_loss_ang = tf.scalar_summary('loss_angular_yaw', evaluation_split[1])

            summary_op = tf.merge_summary([train_loss, train_loss_lin, train_loss_ang,
                train_learning_rate])
            eval_summary_op = tf.merge_summary([eval_loss, eval_loss_lin, eval_loss_ang])

            # Saver for model snapshots
            saver = tf.train.Saver()
        
            self.sess.run(tf.initialize_all_variables())

            # start the tensorflow QueueRunner's
            self.coord = tf.train.Coordinator()
            self.runners = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)
            # start our custom queue runner's threads
            self.custom_data_runner.start_threads(self.sess, self.coord)

            # Save summaries for training and evaluation in separate folders
            summary_writer = tf.train.SummaryWriter(os.path.join(storage_path, 'train'), self.sess.graph)
            eval_summary_writer = tf.train.SummaryWriter(os.path.join(storage_path, 'eval'), self.sess.graph)

            # Save the tensorflow graph definition as protobuf file (does not include weights)
            tf.train.write_graph(self.sess.graph_def, os.path.join(storage_path), 'graph.pb', False) #proto

            # Vector to average the duration over the last report steps
            duration_vector = [0.0] * (self.args.eval_steps // 100)


            if self.args.weight_initialize:

                logger.info('Initialize with weights from another model')

                if os.path.exists(self.args.weight_initialize):
                    saver.restore(self.sess, self.args.weight_initialize)
                    logger.info('Model restored: {}'.format(self.args.weight_initialize))
                else:
                    logger.warning('No weights are loaded!')
                    logger.warning('File does not exist: {}'.format(self.args.weight_initialize))

            logger.info('Load the evaluation data')
            (X_eval,Y_eval) = DataHandler(self.args.datafile_eval, self.eval_n_elements,
                    shuffle=False).next_batch()
            X_eval = self.check_extend(X_eval, np.ceil(self.eval_n_elements / self.eval_batch_size) * self.eval_batch_size)

            loss_train = 0.0
            # Perform all training steps

            logger.info('Training begins')
            for step in range(self.args.max_steps):
                start_time = time.time()

                feed_dict = {keep_prob_placeholder: 0.5}
                _, loss_value, loss_split_value, summary_str = self.sess.run([train_op, loss, loss_split, summary_op],
                        feed_dict=feed_dict)

                duration = time.time() - start_time

                # Report every 100 steps
                if step > 0 and step % 100 == 0:
                    # Print status to stdout.
                    logger.info('Step {}: loss = ({:.4f},{:.4f}) {:.3f} msec'.format(step,
                        loss_split_value[0], loss_split_value[1], duration/1e-3))
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()
                    loss_train = loss_value

                    # Replace the durations in fifo fashion
                    duration_vector[((step % self.args.eval_steps)//100)] = duration

                # Evaluatie the model
                if step > 0 and step % self.args.eval_steps == 0 or step == (self.args.max_steps - 1):
                    start_eval = time.time()

                    # Create an empty array, that has the correct size for to hold all predictions
                    eval_predictions = np.zeros([X_eval.shape[0],2], dtype=np.float)

                    # Evaluate the data in batches and capture the predictions
                    for index in range(X_eval.shape[0] // self.eval_batch_size):
                        start_index = index*self.eval_batch_size
                        end_index = (index+1)*self.eval_batch_size
                        feed_dict = {
                                eval_data_placeholder: X_eval[start_index:end_index,:],
                                keep_prob_placeholder: 1.0
                                }
                        eval_predictions[start_index:end_index,:] = self.sess.run([eval_prediction],
                                feed_dict=feed_dict)[0]

                    # Finally evaluate the predictions and compute the scores
                    feed_dict = {
                            eval_predictions_placeholder: eval_predictions[:self.eval_n_elements,:],
                            eval_cmd_placeholder:  Y_eval,
                            keep_prob_placeholder: 1.0
                            }

                    loss_value, loss_split_value, summary_str = self.sess.run([evaluation,
                        evaluation_split, eval_summary_op], feed_dict=feed_dict)

                    duration_eval = time.time() - start_eval

                    logger.info('Evaluattion: loss = ({:.4f},{:.4f}) {:.3f} msec'.format(loss_split_value[0], loss_split_value[1], duration_eval/1e-3))

                    eval_summary_writer.add_summary(summary_str, step)
                    eval_summary_writer.flush()

                if step > 0 and step % 1000 == 0:
                    # Save a checkpoint
                    logger.info('Save model snapshot')
                    filename = os.path.join(storage_path, 'snap')
                    saver.save(self.sess, filename, global_step=step)

            logger.info('Save final model snapshot')
            filename = os.path.join(storage_path, 'final')
            saver.save(self.sess, filename)

            # Save the model with weights in one file
            # This will only capture the operations used to generate the prediction. It also
            # replaces the variables with the weights from training as constant values 
            # See:
            # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py
            logger.info('Save final model with weights')
            output_node_names = 'model_inference'
            output_graph_def = tf.python.client.graph_util.convert_variables_to_constants(
                    self.sess, self.sess.graph_def, output_node_names.split(","))
            with tf.gfile.GFile(os.path.join(storage_path, 'model.pb'), "wb") as f:
                    f.write(output_graph_def.SerializeToString())
                    logger.info("{} ops in the final graph.".format(len(output_graph_def.node)))

        if self.args.mail:
            self.send_notification(loss_train, loss_value)

    def send_notification(self, loss_train, loss_eval):
        subject = 'Training has finished'
        message = 'Error Training: {}\nError Evaluation: {}\nPlease login to see the results.'.format(loss_train, loss_eval)
        user = getpass.getuser()

        os.system('echo "{}" | mail -s "{}" {}'.format(message, subject, user))
