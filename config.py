class Config(object):
    def __init__(self, is_training=True):
        self.init_scale = 0.1
        self.learning_rate = 1.0
        self.max_grad_norm = 5
        self.num_layers = 2
        self.hidden_size = 200
        self.epoch_start_decay = 10
        self.epoch_num = 30
        self.keep_prob = 0.8
        self.lr_decay = 0.5
        self.vocab_size = 10000
        if is_training:
            self.step_size = 20  # sequence len
            self.batch_size = 20
        else:
            self.step_size = 1
            self.batch_size = 1