import numpy as np
import tensorflow as tf
import reader
import config
from lstm import LSTM

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('test', None, 'Path of test file.')
saver = tf.train.Saver()


def run_epoch(sess, model, data, stage):
    epoch_size = (len(data) // model.batch_size - 1) // model.step_size
    total_cost, iters = 0, 0
    if stage == 'test':
        saver.restore(sess, 'saved_model/lstm-model.ckpt')
    state = sess.run(model.initial_state)
    train_op = model.train_op if stage == 'train' else tf.no_op()
    for step, (x, y) in enumerate(reader.ptb_iterator(data, model.batch_size, model.step_size)):
        cost, state, _ = sess.run(
            [model.cost, model.final_state, train_op],
            feed_dict={model.input_data: x, model.targets: y, model.initial_state: state})
        total_cost += cost
        iters += model.step_size
        perplexity = np.exp(total_cost / iters)
        if stage == 'train' and step % 100 == 0:
            progress = (step *1.0/ epoch_size) * 100
            print(' >> Progress %.2f%%: perplexity = %.3f, cost = %.3f' % (progress, perplexity, cost))
    if stage == 'train':
        save_path = saver.save(sess, 'saved_model/lstm-model.ckpt')
        print('Model saved in %s.' % save_path)
    return perplexity


def main(_):
    path = 'data/ptb'

    if not FLAGS.test:
        train_data, vocab_size = reader.ptb_raw_data(path, stage='train')
        print('Vocabulary size = %d' % vocab_size)
        valid_data = reader.ptb_raw_data(path, stage='valid')
        with tf.Graph().as_default(), tf.Session() as sess:
            initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
            with tf.variable_scope('model', reuse=None, initializer=initializer):
                train_model = PTBModel(config=config.config(), stage='train')
            with tf.variable_scope('model', reuse=True):
                valid_model = PTBModel(config=config.config(), stage='valid')

            sess.run(tf.global_variables_initializer())

            for epoch in range(config.epoch_num):
                lr_decay = config.lr_decay ** max(epoch - config.epoch_start_decay, 0)
                train_model.assign_lr(sess, config.learning_rate * lr_decay)
                train_perplexity = run_epoch(sess, train_model, train_data, stage='train')
                valid_perplexity = run_epoch(sess, valid_model, valid_data, stage='valid')
                print('Epoch %d:' % epoch)
                print(' >> Learning rate = %.3f' % sess.run(train_model.lr))
                print(' >> Train perplexity = %.3f' % train_perplexity)
                print(' >> Valid perplexity = %.3f' % valid_perplexity)
    else:
        test_data, _ = reader.ptb_raw_data(path, stage='test')
        with tf.Graph().as_default(), tf.Session() as sess:
            with tf.variable_scope('model', reuse=None):
                test_model = PTBModel(config=config.test_config, stage='test')

            sess.run(tf.global_variables_initializer())

            test_perplexity = run_epoch(sess, test_model(), test_data, stage='test')
            print('Test Perplexity = %.3f' % test_perplexity)


if __name__ == '__main__':
    tf.app.run()