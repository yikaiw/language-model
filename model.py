import reader
import tensorflow as tf


class PTBModel:
    def __init__(self, is_training, config):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        
        # hidden num for a single LSTM
        hidden_size = config.hidden_size
        vocab_size = config.vocab_size

        # input data x
        self.input_data = tf.placeholder(dtype=tf.int32, shape=(batch_size, num_steps))
        # target data y
        self.targets = tf.placeholder(dtype=tf.int32, shape=(batch_size, num_steps))

        #创建单个LSTM，隐匿层的单元数量，遗忘门的初始值可以为1，三向门为开
        lstm_cell = contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
        
        if is_training and config.keep_prob < 1:
            lstm_cell = contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=config.keep_prob)
        
        cell = contrib.rnn.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)
        
        self.initial_state = cell.zero_state(batch_size, tf.float32)


        with tf.device('/cpu:0'):
            embedding = tf.get_variable(
                name='embedding', shape=(vocab_size, hidden_size), dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, self.input_data)
        
        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, keep_prob=config.keep_prob)
        
        #简单调用实现方式
        # from tensorflow.models.rnn import rnn
        # inputs = [tf.squeeze(input_, [1])
        # for input_ in tf.split(1, num_steps, inputs)]:
        #   outputs, state = rnn.rnn(cell, inputs, initial_state=self.initial_state)

        outputs = []
        
        state = self.initial_state
        
        with tf.variable_scope('RNN'):
            for time_step in range(num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                    
                # 从state开始运行RNN架构，输出为cell的输出以及新的state.
                cell_out, state = cell(inputs[:, time_step, :], state)
                outputs.append(cell_out)
        
        #输出定义为cell的输出乘以softmax weight w后加上softmax bias b. 即logit
        output = tf.reshape(tf.concat(outputs, axis=1), [-1, hidden_size])
        softmax_w = tf.get_variable('softmax_w', (hidden_size, vocab_size),dtype=tf.float32)
        softmax_b = tf.get_variable('softmax_b', (vocab_size,), dtype=tf.float32)
        logits = tf.matmul(output, softmax_w) + softmax_b
        
        #loss函数是average negative log probability, 函数sequence_loss_by_examples实现
        loss = contrib.legacy_seq2seq.sequence_loss_by_example(
            logits=[logits], targets=[tf.reshape(self.targets,[-1])],
            weights=[tf.ones((batch_size * num_steps,),dtype=tf.float32)])        
        self.cost = cost = tf.reduce_sum(loss) / batch_size
        self.final_state = state

        if not is_training:
            return
        
        # learning rate
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        
        # 根据张量间的和的norm来clip多个张量
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)
        
        # 用之前的变量learning rate来起始梯度下降优化器。
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        
        # 一般的minimize为先取compute_gradient,再用apply_gradient
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))
        self.new_lr = tf.placeholder(dtype=tf.float32, shape=[],name='new_learning_rate')
        self.lr_update = tf.assign(self.lr, self.new_lr)
    
    #更新learning rate
    def assignlr(self, session, lr_value):
        session.run(self.lr_update, feed_dict={self.new_lr: lr_value})