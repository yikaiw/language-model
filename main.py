import numpy as np
import tensorflow as tf
import reader
from config import Config
from lstm import LSTM

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_boolean('is_testing', False, 'If is testing, default False.')
saver = tf.train.Saver()
save_path = os.path.join('saved_model', 'lstm-model.ckpt')
data_path = os.path.join('data', 'ptb')


def run_epoch(sess, model, data, is_training=True):
    epoch_size = (len(data) // model.batch_size - 1) // model.step_size
    total_cost, iters = 0, 0
    state = sess.run(model.initial_state)
    op = model.train_op if is_training else tf.no_op()

    for step, (x, y) in enumerate(reader.ptb_iterator(data, model.batch_size, model.step_size)):
        cost, state, _ = sess.run(
            [model.cost, model.final_state, op],
            feed_dict={model.input_data: x, model.targets: y, model.initial_state: state})
        total_cost += cost
        iters += model.step_size
        perplexity = np.exp(total_cost / iters)
        if is_training and step % 100 == 0:
            progress = (step *1.0/ epoch_size) * 100
            print(' >> Progress %.1f%%: perplexity = %.3f, cost = %.3f' % (progress, perplexity, cost))

    return perplexity


def main(_):
    if not FLAGS.is_testing:
        train_data, valid_data = reader.ptb_raw_data(data_path)
        config = Config()

        with tf.Graph().as_default(), tf.Session() as sess:
            initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
            with tf.variable_scope('model', reuse=None, initializer=initializer):
                train_model = LSTM()
            with tf.variable_scope('model', reuse=True):
                valid_model = LSTM(is_training=False)

            sess.run(tf.global_variables_initializer())

            for epoch in range(config.epoch_num):
                lr_decay = config.lr_decay ** max(epoch - config.epoch_start_decay, 0)
                train_model.assign_lr(sess, config.learning_rate * lr_decay)
                train_perplexity = run_epoch(sess, train_model, train_data)
                valid_perplexity = run_epoch(sess, valid_model, valid_data, is_training=False)
                print('Epoch %d:' % epoch)
                print(' >> Learning rate = %.3f' % sess.run(train_model.lr))
                print(' >> Train perplexity = %.3f' % train_perplexity)
                print(' >> Valid perplexity = %.3f' % valid_perplexity)

            saver.save(sess, save_path)
            print('Model saved in %s.' % save_path)

    else:
        test_data, _ = reader.ptb_raw_data(data_path, is_testing=True)

        with tf.Graph().as_default(), tf.Session() as sess:
            saver.restore(sess, save_path)
            with tf.variable_scope('model', reuse=None):
                test_model = LSTM(is_training=False)

            sess.run(tf.global_variables_initializer())

            test_perplexity = run_epoch(sess, test_model(), test_data, is_training=False)
            print('Test Perplexity = %.3f' % test_perplexity)


if __name__ == '__main__':
    tf.app.run()