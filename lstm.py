import reader
import tensorflow as tf
from config import Config


class LSTM(object):
    def __init__(self, config, is_training=True):
        config = Config(is_training=is_training)

        batch_size = self.batch_size = config.batch_size
        step_size = self.step_size = config.step_size
        hidden_size = config.hidden_size
        vocab_size = config.vocab_size

        self.input_data = tf.placeholder(tf.int32, [batch_size, step_size])
        self.targets = tf.placeholder(tf.int32, [batch_size, step_size])

        lstm_cell = contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
        
        if is_training and config.keep_prob < 1:
            lstm_cell = contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=config.keep_prob)
        
        stacked_lstm = contrib.rnn.MultiRNNCell([lstm_cell] * config.layer_num, state_is_tuple=True)
        
        self.initial_state = stacked_lstm.zero_state(batch_size, tf.float32)

        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [vocab_size, hidden_size], tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, self.input_data)
            # shape of inputs: [batch_size, step_size, hidden_size]
        
        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, keep_prob=config.keep_prob)
        
        outputs = []
        
        state = self.initial_state
        
        with tf.variable_scope('RNN'):
            for time_step in range(step_size):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                output, state = stacked_lstm(inputs[:, time_step, :], state)
                outputs.append(output)
        
        output = tf.reshape(tf.concat(outputs, axis=1), [-1, hidden_size])
        softmax_w = tf.get_variable('softmax_w', [hidden_size, vocab_size], dtype=tf.float32)
        softmax_b = tf.get_variable('softmax_b', [vocab_size], dtype=tf.float32)
        logits = tf.matmul(output, softmax_w) + softmax_b
        
        # average negative log probability loss
        loss = contrib.legacy_seq2seq.sequence_loss_by_example(
            logits=[logits], targets=[tf.reshape(self.targets, [-1])],
            weights=[tf.ones([batch_size * step_size], tf.float32)])
        self.cost = tf.reduce_sum(loss) / batch_size
        self.final_state = state

        if is_training:
            self.lr = tf.Variable(0, trainable=False)
            tvars = tf.trainable_variables()
            
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), config.max_grad_norm)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
            
            self.opt = optimizer.apply_gradients(zip(grads, tvars))
            self.new_lr = tf.placeholder(tf.float32)
            self.lr_update = tf.assign(self.lr, self.new_lr)
        else:
            self.opt = tf.no_op()
    
    def assignlr(self, sess, lr_value):
        sess.run(self.lr_update, feed_dict={self.new_lr: lr_value})