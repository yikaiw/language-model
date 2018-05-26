import numpy as np
import tensorflow as tf
import reader
import config
from model import PTBModel


def run_epoch(sess, m, data, eval_op, verbose=False):
    epoch_size = (len(data) // m.batch_size - 1) // m.step_size
    costs, iters = 0, 0
    state = sess.run(m.initial_state)
    for step, (x, y) in enumerate(reader.ptb_iterator(data, m.batch_size, m.step_size)):
        cost, state, _ = sess.run(
            [m.cost, m.final_state, eval_op],
            feed_dict={m.input_data: x, m.targets: y, m.initial_state: state})
        costs += cost
        iters += m.step_size
        if verbose and step % (epoch_size // 10) == 10:
            print('%.3f perplexity: %.3f' % (step * 1.0 / epoch_size, np.exp(costs / iters)))
    return np.exp(costs / iters)


def main():
    path = 'data/ptb'
    raw_data = reader.ptb_raw_data(path)
    train_data, valid_data, test_data, _ = raw_data
    
    with tf.Graph().as_default(), tf.Session() as sess:
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
        
        with tf.variable_scope('model', reuse=None, initializer=initializer):
            m_train = PTBModel(is_training=True)
        
        with tf.variable_scope('model', reuse=True, initializer=initializer):
            m_valid = PTBModel(is_training=False)
            m_test = PTBModel(is_training=False, is_testing=True)

        sess.run(tf.global_variables_initializer())
        
        for epoch in range(config.epoch_num):
            lr_decay = config.lr_decay ** max(epoch - config.epoch_start_decay, 0)
            m_train.assign_lr(sess, config.learning_rate * lr_decay)
            print('Epoch: %d Learning rate: %.3f' % (epoch + 1, sess.run(m_train.lr)))
            train_perplexity = run_epoch(sess, m_train, train_data, m_train.train_op, verbose=True)
            print('Epoch: %d Train Perplexity: %.3f' % (epoch + 1, train_perplexity))
            valid_perplexity = run_epoch(sess, m_valid, valid_data, tf.no_op())
            print('Epoch: %d Valid Perplexity: %.3f' % (epoch + 1, valid_perplexity))

        test_perplexity = run_epoch(sess, m_test, test_data, tf.no_op())
        print('Test Perplexity: %.3f' % test_perplexity)


if __name__ == '__main__':
    tf.app.run()