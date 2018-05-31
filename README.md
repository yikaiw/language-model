# language-model
Language model with LSTM.

## Guidance
* Train: run **python main.py**, the loss and perplexity will be printed in the screen during training. After training, run **tensorboard --logdir=logs** to see the curves of training loss and training perplexity. The model will be saved in models/.

* Test: Put the test file named 'test.txt' into the director data/ptb/, and run **python main.py --testing True** to test a well-trained model saved in the path saved_model/.

## Description
* The network designed in lstm.py has 2 layers of LSTM. Each LSTM layer is wrapped with a dropout layer. Each word is converted to an embedding with 200 dimensions, as well as the hidden dimensions of the LSTM. The embeddings also need to pass a dropout layer. 

* The memory state of the network is initialized with zero. In order to ovoid gradient vanish, make the learning process tractable, I use an unrolled version of network which contains a fixed length of words, i.e. step_size, to feed in LSTM's inputs and targets. 

* The loss function of the network is negative log probability loss. Perplexity equals to average negative log probability loss square e.

* I use a decaying learning rate with 0.7 decay rate during the training, and train for 20 epochs. After finishing an epoch, I cross validate the current model using data valid.txt.

## Results
* After 45k step, the training loss is around 90, the training perplexity is around 70, and validation perplexity is around 102.

