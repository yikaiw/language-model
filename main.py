import numpy as np
import tensorflow as tf
import reader
import config
from lstm import LSTM

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('test', None, 'Path of test file.')


def run_epoch(sess, model, data, is_training):
    epoch_size = (len(data) // model.batch_size - 1) // model.step_size
    total_cost, iters = 0, 0
    saver = tf.train.Saver()
    if not is_training:
        saver.restore(sess, 'saved_model/lstm-model.ckpt')
    state = sess.run(model.initial_state)
    train_op = model.train_op if is_training else tf.no_op()
    for step, (x, y) in enumerate(reader.ptb_iterator(data, model.batch_size, model.step_size)):
        cost, state, _ = sess.run(
            [model.cost, model.final_state, train_op],
            feed_dict={model.input_data: x, model.targets: y, model.initial_state: state})
        total_cost += cost
        iters += model.step_size
        perplexity = np.exp(total_cost / iters)
        if is_training and step % 100 == 0:
            progress = (step *1.0/ epoch_size) * 100
            print(' >> Progress %.2f%%: perplexity = %.3f, cost = %.3f' % (progress, perplexity, cost))
    if is_training:
        save_path = saver.save(sess, 'saved_model/lstm-model.ckpt')
        print('Model saved in %s.' % save_path)
    return perplexity


def main(_):
    path = 'data/ptb'
    raw_data = reader.ptb_raw_data(path)

    if not FLAGS.test:
        train_data, valid_data, _ = raw_data(is_training=True)
        with tf.Graph().as_default(), tf.Session() as sess:
            initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
            with tf.variable_scope('model', reuse=None, initializer=initializer):
                m_train = PTBModel(is_training=True)
            with tf.variable_scope('model', reuse=True, initializer=initializer):
                m_valid = PTBModel(is_training=False)

            sess.run(tf.global_variables_initializer())

            for epoch in range(config.epoch_num):
                lr_decay = config.lr_decay ** max(epoch - config.epoch_start_decay, 0)
                m_train.assign_lr(sess, config.learning_rate * lr_decay)
                train_perplexity = run_epoch(sess, m_train, train_data, is_training=True)
                valid_perplexity = run_epoch(sess, m_valid, valid_data, is_training=False)
                print('Epoch %d:' % epoch)
                print(' >> Learning rate = %.3f' % sess.run(m_train.lr))
                print(' >> Train perplexity = %.3f' % train_perplexity)
                print(' >> Valid perplexity = %.3f' % train_valid_perplexityperplexity)
    else:
        test_data, _ = raw_data(is_training=False)
        with tf.Graph().as_default(), tf.Session() as sess:
            initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
            with tf.variable_scope('model', reuse=True, initializer=initializer):
                m_test = PTBModel(is_training=False, is_testing=True)

            sess.run(tf.global_variables_initializer())

            test_perplexity = run_epoch(sess, m_test, test_data, is_training=False)
            print('Test Perplexity = %.3f' % test_perplexity)


if __name__ == '__main__':
    tf.app.run()