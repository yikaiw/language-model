import os
import sys
import numpy as np
import tensorflow as tf
import collections


def _read_words(filename):
    with tf.gfile.GFile(filename, 'r') as f:
        return f.read().decode('utf-8').replace('\n', '<eos>').split()


def _build_vocab(filename):
    data = _read_words(filename)
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    return word_to_id


def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


def ptb_raw_data(is_training, data_path=None):
    '''Load PTB raw data from data directory "data_path".
    Reads PTB text files, converts strings to integer ids, and performs mini-batching of the inputs.
    The PTB dataset comes from Tomas Mikolov's webpage:
    http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
    Args:
        is_training: if is training or not
        data_path: string path to the directory where simple-examples.tgz has been extracted.
    Returns:
        tuple (train_data, valid_data, test_data, vocabulary)
        where each of the data objects can be passed to PTBIterator.
    '''
    train_path = os.path.join(data_path, 'train.txt')
    word_to_id = _build_vocab(train_path)
    vocabulary = len(word_to_id)
    if is_training:
        valid_path = os.path.join(data_path, 'valid.txt')
        train_data = _file_to_word_ids(train_path, word_to_id)
        valid_data = _file_to_word_ids(valid_path, word_to_id)
        return train_data, valid_data, vocabulary
    else:
        test_path = os.path.join(data_path, 'test.txt')
        test_data = _file_to_word_ids(test_path, word_to_id)
        return test_data, vocabulary


def ptb_producer(raw_data, batch_size, step_size, name=None):
    '''Iterate on the raw PTB data.
    This chunks up raw_data into batches of examples and returns Tensors that are drawn from these batches.
    Args:
        raw_data: one of the raw data outputs from ptb_raw_data.
        batch_size: int, the batch size.
        step_size: int, the number of unrolls.
        name: the name of this operation (optional).
    Returns:
        A pair of Tensors, each shaped [batch_size, step_size]. The second element
        of the tuple is the same data time-shifted to the right by one.
    Raises:
        tf.errors.InvalidArgumentError: if batch_size or step_size are too high.
    '''
    with tf.name_scope(name, 'PTBProducer', [raw_data, batch_size, step_size]):
        raw_data = tf.convert_to_tensor(raw_data, name='raw_data', dtype=tf.int32)
        data_len = tf.size(raw_data)
        batch_len = data_len // batch_size
        data = tf.reshape(raw_data[0 : batch_size * batch_len], [batch_size, batch_len])
        epoch_size = (batch_len - 1) // step_size
        assertion = tf.assert_positive(
            epoch_size,
            message='epoch_size == 0, decrease batch_size or step_size')
        with tf.control_dependencies([assertion]):
            epoch_size = tf.identity(epoch_size, name='epoch_size')
        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = tf.strided_slice(data, [0, i * step_size], [batch_size, (i + 1) * step_size])
        x.set_shape([batch_size, step_size])
        y = tf.strided_slice(data, [0, i * step_size + 1], [batch_size, (i + 1) * step_size + 1])
        y.set_shape([batch_size, step_size])
        return x, y


def ptb_iterator(raw_data, batch_size, step_size):
    raw_data = np.array(raw_data, dtype=np.int32)
    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i: batch_len * (i + 1)]
    epoch_size = (batch_len - 1) // step_size
    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or step_size")
    for i in range(epoch_size):
        x = data[:, i * step_size: (i + 1) * step_size]
        y = data[:, i * step_size + 1: (i + 1) * step_size + 1]
        yield (x, y)


# Test reader
if __name__ == '__main__':
    path = 'data/reader_test'
    raw_data = ptb_raw_data(path)
    train_data, valid_data, test_data, _ = raw_data
    print(train_data)
    x, y = ptb_iterator(valid_data, 3, 2)
    print(x, y)
