
import _init_paths

import argparse
import logging
import os
import time

import tensorflow as tf

from data.data_handler import DataHandler
from data.fast_data_handler import FastDataHandler
import model.model as model

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Train our model on the given dataset')

    def check_extension(extensions, filename):
        ext = os.path.splitext(filename)[1]
        if ext not in extensions:
            parser.error("Unsupported file extension: Use {}".format(extensions))
        return filename

    parser.add_argument('datafile_train', help='Filename of the training data', type=lambda
            s:check_extension(['.h5'],s))
    parser.add_argument('datafile_eval', help='Filename of the evaluation data', type=lambda
            s:check_extension(['.h5'],s))
    parser.add_argument('-s', '--max_steps', help='Number of batches to run', type=int, default=1000000)
    parser.add_argument('-b', '--batch_size', help='Size of training batches', type=int,  default=16)
    parser.add_argument('-t', '--train_dir', help='Directory to save the model snapshots',
            default='./models/default')
    parser.add_argument('-l', '--learning_rate', help='Initial learning rate', type=float, default=0.01)
    args = parser.parse_args()

    return args

def placeholder_inputs(data_size, batch_size):
    """Create placeholders for the tf graph"""
    data_placeholder = tf.placeholder(tf.float32, shape=None)
    cmd_placeholder = tf.placeholder(tf.float32, shape=None)

    return data_placeholder, cmd_placeholder

def run_training(args):
    """Train a tensorflow model"""
    logger = logging.getLogger(__name__)

    storage_path = os.path.join(args.train_dir, (time.strftime('%Y-%m-%d_%H-%M_') + model.NAME))

    with tf.Graph().as_default():

        learning_rate = tf.Variable(args.learning_rate, trainable=False)

        data_handler_train = FastDataHandler(args.datafile_train, args.batch_size)

        data_placeholder, cmd_placeholder = placeholder_inputs(model.INPUT_SIZE, args.batch_size)

        prediction = model.inference(data_placeholder)

        loss, loss_split = model.loss(prediction, cmd_placeholder)

        train_op = model.training(loss, loss_split, learning_rate)

        evaluation, evaluation_split = model.evaluation(prediction, cmd_placeholder)

        tf.scalar_summary('loss', loss)
        tf.scalar_summary('loss_linear_x', loss_split[0])
        tf.scalar_summary('loss_angular_yaw', loss_split[1])
        tf.scalar_summary('learning_rate', learning_rate)

        summary_op = tf.merge_all_summaries()
        eval_summary_op = tf.merge_all_summaries()

        saver = tf.train.Saver()
            
        with tf.Session() as sess:
        
            sess.run(tf.initialize_all_variables())

            summary_writer = tf.train.SummaryWriter(os.path.join(storage_path, 'train'), sess.graph)
            eval_summary_writer = tf.train.SummaryWriter(os.path.join(storage_path, 'eval'), sess.graph)

            for step in range(args.max_steps):
                start_time = time.time()

                (X,Y) = data_handler_train.next_batch()

                feed_dict = {data_placeholder: X, cmd_placeholder: Y}

                _, loss_value, loss_split_value = sess.run([train_op, loss, loss_split], feed_dict=feed_dict)

                duration = time.time() - start_time

                if step > 0 and step % 100 == 0:
                    # Print status to stdout.
                    logger.info('Step {}: loss = ({:.4f},{:.4f}) {:.3f} msec'.format(step,
                        loss_split_value[0], loss_split_value[1], duration/1e-3))
                    summary_str = sess.run(summary_op, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()

                if step > 0 and step % 1000 == 0 or step == args.max_steps:
                    # Evaluate model
                    logger.info('Evaluate model')
                    evaluate_model(sess, evaluation, evaluation_split, data_placeholder,
                            cmd_placeholder, eval_summary_op, step, args.datafile_eval,
                            4*8192, writer=eval_summary_writer)

                if step > 0 and step % 1000 == 0 or step == args.max_steps:
                    # Save a checkpoint
                    logger.info('Save model snapshot')
                    filename = os.path.join(storage_path, 'snap')
                    saver.save(sess, filename, global_step=step)

def evaluate_model(sess, evaluation, evaluation_split, data_placeholder, cmd_placeholder,
        summary_op, step, datafile_eval, eval_n_elements, writer=None):

    data_handler_eval = FastDataHandler(datafile_eval, eval_n_elements, eval_n_elements)

    (X,Y) = data_handler_eval.next_batch()

    feed_dict = {data_placeholder: X, cmd_placeholder: Y}
    loss_value, loss_split_value = sess.run([evaluation, evaluation_split], feed_dict=feed_dict)

    summary_str = sess.run(summary_op, feed_dict=feed_dict)
    writer.add_summary(summary_str, step)
    writer.flush()

def main():
    logger = logging.getLogger(__name__)
    logger.info('Train our model on the given dataset:')

    args = parse_args()
    logger.info(args.datafile_train)

    # Open data handler
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    logger.info('Start training')
    run_training(args)

if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
