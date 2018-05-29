import tensorflow as tf
from tensorflow.contrib import rnn
from datetime import datetime
from config import Config
import reader

current_time = datetime.now().strftime('%Y%m%d-%H%M')        


class LSTM(object):
    def __init__(self, sess, stage='train'):
        self.sess = sess
        self.stage = stage
        config = Config(stage == 'test')

        batch_size = self.batch_size = config.batch_size
        step_size = self.step_size = config.step_size
        hidden_size = config.hidden_size
        vocab_size = config.vocab_size

        self.input_data = tf.placeholder(tf.int32, [batch_size, step_size])
        self.targets = tf.placeholder(tf.int32, [batch_size, step_size])

        lstm_cell = rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
        
        if stage == 'train' and config.keep_prob < 1:
            lstm_cell = rnn.DropoutWrapper(lstm_cell, output_keep_prob=config.keep_prob)

        # lstm_cell = rnn.AttentionCellWrapper(lstm_cell, attn_length=5)
        
        stacked_lstm = rnn.MultiRNNCell([lstm_cell] * config.layer_num, state_is_tuple=True)
        
        self.initial_state = stacked_lstm.zero_state(batch_size, tf.float32)

        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [vocab_size, hidden_size], tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, self.input_data)
            # shape of inputs: [batch_size, step_size, hidden_size]
        
        if stage == 'train' and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, keep_prob=config.keep_prob)
        
        outputs = []
        
        state = self.initial_state
        
        with tf.variable_scope('rnn'):
            for time_step in range(step_size):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                output, state = stacked_lstm(inputs[:, time_step, :], state)
                outputs.append(output)
            
        self.final_state = state
        
        outputs = tf.reshape(tf.concat(outputs, axis=1), [-1, hidden_size])
        softmax_w = tf.get_variable('softmax_w', [hidden_size, vocab_size], dtype=tf.float32)
        softmax_b = tf.get_variable('softmax_b', [vocab_size], dtype=tf.float32)
        logits = tf.matmul(outputs, softmax_w) + softmax_b  # [batch_size * step_size, vocab_size]
        targets = tf.reshape(self.targets, [-1])  # [batch_size * step_size]
        
        self.loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            logits=[logits], targets=[targets], 
            weights=[tf.ones([batch_size * step_size], tf.float32)])
        self.loss = tf.reduce_sum(self.loss) / batch_size
        # self.loss = - sum_i^N(log P(target_i)), needs to / N outside -> - 1 / N * sum_i^N(log P(target_i))

        self.init_summary()

        if stage == 'train':
            self.lr = tf.Variable(0, trainable=False, dtype=tf.float32)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), config.max_grad_norm)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
            self.opt = optimizer.apply_gradients(zip(grads, tvars))
            self.new_lr = tf.placeholder(tf.float32)
            self.lr_update = tf.assign(self.lr, self.new_lr)
        else:
            self.opt = tf.no_op()
            
    def init_summary(self):
        if self.stage == 'train':
            self.loss_writer = tf.summary.FileWriter('logs/loss@%s' % current_time, self.sess.graph)
            self.perplexity_writer = tf.summary.FileWriter('logs/perplexity@%s' % current_time, self.sess.graph)
            self.perplexity = tf.placeholder(tf.float32)
            tf.add_to_collection('perplexity', tf.summary.scalar('perplexity', self.perplexity))
            tf.add_to_collection('loss', tf.summary.scalar('loss', self.loss))
            self.summary_loss = tf.summary.merge_all('loss')
            self.summary_perplexity = tf.summary.merge_all('perplexity')
        else:
            self.summary_loss = tf.no_op()
            
    def assign_lr(self, lr_value):
        self.sess.run(self.lr_update, feed_dict={self.new_lr: lr_value})