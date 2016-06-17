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
    data_placeholder = tf.placeholder(tf.float32, shape=[None, 723], name='data_input')
    cmd_placeholder = tf.placeholder(tf.float32, shape=[None, 2])

    return data_placeholder, cmd_placeholder

def run_training(args):
    """Train a tensorflow model"""
    logger = logging.getLogger(__name__)

    storage_path = os.path.join(args.train_dir, (time.strftime('%Y-%m-%d_%H-%M_') + model.NAME))

    with tf.Graph().as_default():

        global_step, learning_rate = model.learning_rate(args.learning_rate)

        data_handler_train = FastDataHandler(args.datafile_train, args.batch_size, 2**18)

        data_placeholder, cmd_placeholder = placeholder_inputs(model.INPUT_SIZE, args.batch_size)

        prediction = model.inference(data_placeholder)

        loss, loss_split = model.loss(prediction, cmd_placeholder)

        train_op = model.training(loss, loss_split, learning_rate, global_step)

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

            tf.train.write_graph(sess.graph_def, os.path.join(storage_path), "graph.pb", False) #proto

            for step in range(args.max_steps):
                start_time = time.time()

                (X,Y) = data_handler_train.next_batch()

                load_duration = time.time() - start_time

                feed_dict = {data_placeholder: X, cmd_placeholder: Y}

                _, loss_value, loss_split_value = sess.run([train_op, loss, loss_split], feed_dict=feed_dict)

                duration = time.time() - start_time

                if step > 0 and step % 100 == 0:
                    # Print status to stdout.
                    logger.info('Step {}: loss = ({:.4f},{:.4f}) {:.3f} msec (load: {:.3f} msec)'.format(step,
                        loss_split_value[0], loss_split_value[1], duration/1e-3, load_duration/1e-3))
                    summary_str = sess.run(summary_op, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()

                if step > 0 and step % 1000 == 0 or step == args.max_steps:
                    # Evaluate model
                    logger.info('Evaluate model')
                    evaluate_model(sess, evaluation, evaluation_split, data_placeholder,
                            cmd_placeholder, eval_summary_op, args.datafile_eval,
                            4*8192)
                    eval_summary_writer.add_summary(summary_str, step)
                    eval_summary_writer.flush()

                if step > 0 and step % 1000 == 0:
                    # Save a checkpoint
                    logger.info('Save model snapshot')
                    filename = os.path.join(storage_path, 'snap')
                    saver.save(sess, filename, global_step=step)

            logger.info('Save final model snapshot')
            filename = os.path.join(storage_path, 'final')
            saver.save(sess, filename)

            # Save the model with weights in one file
            logger.info('Save final model with weights')
            output_node_names = 'prediction'
            output_graph_def = tf.python.client.graph_util.convert_variables_to_constants(
                    sess, sess.graph_def, output_node_names.split(","))
            with tf.gfile.GFile(os.path.join(storage_path, 'model.pb'), "wb") as f:
                    f.write(output_graph_def.SerializeToString())
                    logger.info("{} ops in the final graph.".format(len(output_graph_def.node)))

def evaluate_model(sess, evaluation, evaluation_split, data_placeholder, cmd_placeholder,
        summary_op, datafile_eval, eval_n_elements):

    data_handler_eval = FastDataHandler(datafile_eval, eval_n_elements, eval_n_elements)

    (X,Y) = data_handler_eval.next_batch()

    feed_dict = {data_placeholder: X, cmd_placeholder: Y}
    loss_value, loss_split_value = sess.run([evaluation, evaluation_split], feed_dict=feed_dict)

    return sess.run(summary_op, feed_dict=feed_dict)

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
