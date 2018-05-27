class config(object):
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    hidden_size = 200
    epoch_start_decay = 10
    epoch_num = 30
    keep_prob = 0.8
    lr_decay = 0.5
    vocab_size = 10000

    step_size = 20  # sequence len
    batch_size = 20
    

class test_config(object):
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    hidden_size = 200
    epoch_start_decay = 10
    epoch_num = 30
    keep_prob = 0.8
    lr_decay = 0.5
    vocab_size = 10000

    step_size = 1
    batch_size = 1
    