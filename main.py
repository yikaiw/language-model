import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
import reader
from get_config import get_config
from model import PTBModel

# 在函数传递入的session里运行rnn图的cost和 fina_state结果，另外也计算eval_op的结果
# 这里eval_op是作为该函数的输入
def run_iter(session, m, data, eval_op, x, y, state, verbose, step,
             epoch_size, costs, iters, start_time):
    cost, state, _ = session.run([m.cost, m.final_state, eval_op],
                                 feed_dict={m.input_data: x,
                                            m.targets: y,
                                            m.initial_state: state})
    costs += cost
    iters += m.num_steps
    
    # 每一定量运行后输出目前结果
    if verbose and step % (epoch_size // 10) == 10:
        print('%.3f perplexity: %.3f speed: %.0f wps' %
              (step * 1.0 / epoch_size, np.exp(costs / iters),
               iters * m.batch_size / (time.time() - start_time)))
    return costs, iters


def run_epoch(session, m, data, eval_op, verbose=False):
    epoch_size = (len(data) // m.batch_size - 1) // m.num_steps
    start_time = time.time()
    costs = 0.0
    iters = 0.0
    state = session.run(fetches=m.initial_state)
    
    #ptb_iterator函数在接受了输入，batch size以及运行的step数后输出
    #步骤数以及每一步骤所对应的一对x和y的batch数据，大小为[batch_size, num_step]
    for step, (x, y) in enumerate(reader.ptb_iterator(data, m.batch_size, m.num_steps)):
        costs, iters = run_iter(session, m, data, eval_op, x, y, state, verbose, step,
                                epoch_size, costs, iters, start_time)
        
    return np.exp(costs / iters)


def main():
    path = 'data/ptb'
    raw_data = reader.ptb_raw_data(path)
    train_data, valid_data, test_data, _ = raw_data
    
    config = get_config()
    test_config = get_config()
    test_config.batch_size = 1
    test_config.num_steps = 1
    
    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-config.init_scale,config.init_scale)
        
        # train
        with tf.variable_scope('model', reuse=None, initializer=initializer):
            m_train = PTBModel(is_training=True, config=config)
        
        # valid, test
        with tf.variable_scope('model', reuse=True, initializer=initializer):
            m_valid = PTBModel(is_training=False, config=config)
            m_test = PTBModel(is_training=False, config=test_config)

        session.run(tf.global_variables_initializer())
        
        # learning rate decay
        for i in range(config.max_max_epoch):
            lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
            m.assign_lr(session, config.learning_rate * lr_decay)

            print('Epoch: %d Learning rate: %.3f' % (i + 1, session.run(m.lr)))
            
            # print perplexity
            train_perplexity = run_epoch(session, m_train, train_data, m.train_op, verbose=True)
            print('Epoch: %d Train Perplexity: %.3f' % (i + 1, train_perplexity))
            valid_perplexity = run_epoch(session, m_valid, valid_data, tf.no_op())
            print('Epoch: %d Valid Perplexity: %.3f' % (i + 1, valid_perplexity))

        test_perplexity = run_epoch(session, m_test, test_data, tf.no_op())
        print('Test Perplexity: %.3f' % test_perplexity)


if __name__ == '__main__':
    tf.app.run()