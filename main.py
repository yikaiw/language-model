import numpy as np
import tensorflow as tf
import os
from datetime import datetime
import reader
from config import Config
from lstm import LSTM

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_boolean('testing', False, 'If is testing, default False.')

save_path = os.path.join('models', 'lstm.ckpt')
data_path = os.path.join('data', 'ptb')
global_step = 0


def run_epoch(sess, model, data, stage='train', writer=None, summary_op=None):
    epoch_size = (len(data) // model.batch_size - 1) // model.step_size
    total_loss, iters = 0, 0
    state = sess.run(model.initial_state)
    ptb_iterator = reader.ptb_iterator(data, model.batch_size, model.step_size)

    for step, (x, y) in enumerate(ptb_iterator):
        loss, state, _, summary = sess.run(
            [model.loss, model.final_state, model.opt, summary_op],
            feed_dict={model.input_data: x, model.targets: y, model.initial_state: state})
        total_loss += loss
        iters += model.step_size
        # perplexity = np.exp(total_loss / iters)
        tf.summary.scalar('perplexity', perplexity)
        progress = step / epoch_size * 100
        if state != 'test':
            writer.add_summary(summary, global_step + step)
        if stage == 'train' and (step + 1) % 150 == 0:
            print(' >> Train progress %4.1f%%: perplexity = %.3f, loss = %.3f' % (progress, perplexity, loss))
        elif stage == 'test':
            print('\rProgress = %.1f%%' % progress, end='')

    global_step += step
    return perplexity


def main(_):
    if not FLAGS.testing:
        train_data, valid_data = reader.ptb_raw_data(data_path)
        config = Config()

        with tf.Graph().as_default(), tf.Session() as sess:
            print('Building graph.')
            initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
            with tf.variable_scope('model', reuse=None, initializer=initializer):
                train_model = LSTM(stage='train')
            with tf.variable_scope('model', reuse=True):
                valid_model = LSTM(stage='valid')

            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            current_time = datetime.now().strftime('%Y%m%d-%H%M')
            train_writer = tf.summary.FileWriter('logs/train@' + current_time, sess.graph)
            valid_writer = tf.summary.FileWriter('logs/val@' + current_time, sess.graph)
            summary_op = tf.summary.merge_all()

            for epoch in range(config.epoch_num):
                lr_decay = config.lr_decay ** max(epoch - config.epoch_start_decay, 0)
                train_model.assign_lr(sess, config.learning_rate * lr_decay)
                print('Epoch %d/%d:' % (epoch, config.epoch_num))
                print(' >> Learning rate = %.5f' % sess.run(train_model.lr))
                train_perplexity = run_epoch(
                    sess, train_model, train_data, stage='train', writer=train_writer, summary_op=summary_op)
                print(' >> Train perplexity = %.3f' % train_perplexity)
                valid_perplexity = run_epoch(
                    sess, valid_model, valid_data, stage='valid', writer=valid_writer, summary_op=summary_op)
                print(' >> Valid perplexity = %.3f' % valid_perplexity)

            saver.save(sess, save_path)
            print('Model saved in %s.' % save_path)

    else:
        test_data = reader.ptb_raw_data(data_path, is_testing=True)

        with tf.Graph().as_default(), tf.Session() as sess:
            with tf.variable_scope('model', reuse=None):
                test_model = LSTM(stage='test')

            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, save_path)

            test_perplexity = run_epoch(sess, test_model, test_data, stage='test')
            print('\nTest perplexity = %.3f' % test_perplexity)


if __name__ == '__main__':
    tf.app.run()