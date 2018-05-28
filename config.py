class Config(object):
    def __init__(self, is_testing=False):
        self.init_scale = 0.1
        self.max_grad_norm = 5

        self.layer_num = 2
        self.hidden_size = 200  # hidden num for a single LSTM
        self.epoch_num = 30
        self.keep_prob = 0.8
        self.vocab_size = 10000

        self.learning_rate = 1.0
        self.lr_decay = 0.5
        self.epoch_start_decay = 10

        if is_testing:
            self.step_size = 1 
            self.batch_size = 1
        else:
            self.step_size = 20  # sequence len
            self.batch_size = 20