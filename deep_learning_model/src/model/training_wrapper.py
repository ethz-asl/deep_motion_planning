import _init_paths

import logging 
import os
import time

import tensorflow as tf

from data.custom_data_runner import CustomDataRunner
from data.fast_data_handler import FastDataHandler
import model

class TrainingWrapper():
    """Wrap the training"""
    def __init__(self, args):
        self.args = args
        self.coord = None
        self.runners = None
        self.sess = None
        self.custom_data_runner = None

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

    def placeholder_inputs(self, data_size, cmd_size):
        """Create placeholders for the tf graph"""
        data_placeholder = tf.placeholder(tf.float32, shape=[None, data_size], name='data_input')
        cmd_placeholder = tf.placeholder(tf.float32, shape=[None, cmd_size])

        return data_placeholder, cmd_placeholder

    def run(self):
        logger = logging.getLogger(__name__)

        # Folder where to store snapshots, meta data and the final model
        storage_path = os.path.join(self.args.train_dir, (time.strftime('%Y-%m-%d_%H-%M_') + model.NAME))

        with tf.Graph().as_default():

            # Define the used machine learning model
            global_step, learning_rate = model.learning_rate(self.args.learning_rate)

            self.sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=8))
            self.custom_data_runner =  CustomDataRunner(self.args.datafile_train, self.args.batch_size, 2**18)
            data_batch, cmd_batch = self.custom_data_runner.get_inputs()

            prediction = model.inference(data_batch, output_name='prediction')

            loss, loss_split = model.loss(prediction, cmd_batch)

            train_op = model.training(loss, loss_split, learning_rate, global_step)

            eval_data_placeholder, eval_cmd_placeholder = self.placeholder_inputs(723, 2)
            eval_prediction = model.inference(eval_data_placeholder, reuse=True,
                    output_name='eval_prediction')
            evaluation, evaluation_split = model.evaluation(eval_prediction, eval_cmd_placeholder)

            # Variables to use in the summary (shown in tensorboard)
            tf.scalar_summary('loss', loss)
            tf.scalar_summary('loss_linear_x', loss_split[0])
            tf.scalar_summary('loss_angular_yaw', loss_split[1])
            tf.scalar_summary('learning_rate', learning_rate)

            summary_op = tf.merge_all_summaries()
            eval_summary_op = tf.merge_all_summaries()

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

                if os.path.exists(self.args.weight_initialize):
                    saver.restore(self.sess, self.args.weight_initialize)
                    logger.info('Model restored: {}'.format(self.args.weight_initialize))
                else:
                    logger.warning('No weights are loaded!')
                    logger.warning('File does not exist: {}'.format(self.args.weight_initialize))

            eval_n_elements = 4*8192 
            with FastDataHandler(self.args.datafile_eval, eval_n_elements, eval_n_elements) as data_handler_eval:
                (X_eval,Y_eval) = data_handler_eval.next_batch()

            # Perform all training steps
            for step in range(self.args.max_steps):
                start_time = time.time()

                _, loss_value, loss_split_value = self.sess.run([train_op, loss, loss_split])

                duration = time.time() - start_time

                # Report every 100 steps
                if step > 0 and step % 100 == 0:
                    # Print status to stdout.
                    logger.info('Step {}: loss = ({:.4f},{:.4f}) {:.3f} msec'.format(step,
                        loss_split_value[0], loss_split_value[1], duration/1e-3))
                    summary_str = self.sess.run(summary_op)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()

                    # Replace the durations in fifo fashion
                    duration_vector[((step % self.args.eval_steps)//100)] = duration

                # Evaluatie the model
                if step > 0 and step % self.args.eval_steps == 0 or step == self.args.max_steps:
                    start_eval = time.time()

                    # Evaluate the model. We use only a constant fraction of the entire dataset to
                    # reduce the computation time, yet get a rough estimate of the model's
                    # generalization performance
                    feed_dict = {eval_data_placeholder: X_eval, eval_cmd_placeholder: Y_eval}
                    loss_value, loss_split_value = self.sess.run([evaluation, evaluation_split], feed_dict=feed_dict)

                    summary_str, loss_split_value = self.sess.run(summary_op, feed_dict=feed_dict), loss_split_value

                    duration_eval = time.time() - start_eval
                    logger.info('Evaluattion: loss = ({:.4f},{:.4f}) {:.3f} msec'.format(loss_split_value[0], loss_split_value[1], duration_eval/1e-3))

                    eval_summary_writer.add_summary(summary_str, step)
                    eval_summary_writer.flush()

                    # Estimate the time left from the mean durations
                    # remaining_steps = self.args.max_steps - step
                    # combined_duration = np.mean(duration_vector)*remaining_steps + duration_eval * remaining_steps // self.args.eval_steps
                    # m,s = divmod(combined_duration,60)
                    # h,m = divmod(m,60)
                    # d,h = divmod(h, 24)
                    # logger.info('Time left: {:02.0f}:{:02.0f}:{:02.0f}:{:02.0f} (days:hours:minutes:seconds)'.format(d,h,m,s))

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
            output_node_names = 'eval_prediction'
            output_graph_def = tf.python.client.graph_util.convert_variables_to_constants(
                    self.sess, self.sess.graph_def, output_node_names.split(","))
            with tf.gfile.GFile(os.path.join(storage_path, 'model.pb'), "wb") as f:
                    f.write(output_graph_def.SerializeToString())
                    logger.info("{} ops in the final graph.".format(len(output_graph_def.node)))
