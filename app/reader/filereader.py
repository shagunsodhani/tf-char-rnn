from __future__ import division
from __future__ import print_function

import cPickle
import collections
import os

import numpy as np
import tensorflow as tf

vocab_file_name = "vocab.pkl"
data_file_name = "data.pkl"


def _read_file(file_path):
    with tf.gfile.GFile(file_path) as f:
        return f.read()


def _build_vocab(raw_data):
    counter = collections.Counter(raw_data)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    chars, _ = zip(*count_pairs)
    char_to_id = dict(zip(chars, range(len(chars))))
    return char_to_id, chars


def _raw_data_to_char_ids(raw_data, char_to_id):
    return np.array([char_to_id[char] for char in raw_data], dtype=np.int32)


def read_raw_data(input_dir):
    train_path = os.path.join(input_dir, "data.txt")
    raw_data = _read_file(train_path)
    char_to_id, chars = _build_vocab(raw_data)
    train_data = _raw_data_to_char_ids(raw_data, char_to_id)
    return train_data, char_to_id, chars

def batch_iterator(train_data, batch_size, seq_length):
    train_data_len = len(train_data)
    num_batch = int(train_data_len // (batch_size * seq_length))
    if num_batch == 0:
        raise ValueError("num_batch = 0, decrease batch_size or seq_length")
    train_data = train_data[:num_batch * batch_size * seq_length]
    _x_train_data = train_data
    _y_train_data = np.copy(train_data)
    _y_train_data[:-1] = _x_train_data[1:]
    _y_train_data[-1] = _x_train_data[0]
    x_train_data = np.split(_x_train_data.reshape(batch_size, -1), num_batch, 1)
    y_train_data = np.split(_y_train_data.reshape(batch_size, -1), num_batch, 1)
    for i in xrange(len(x_train_data)):
        yield (x_train_data[i], y_train_data[i])


def persistVocab(vocab, dir_to_save):
    file_path = os.path.join(dir_to_save, vocab_file_name)
    with open(file_path, 'wb') as f:
        cPickle.dump(vocab, f)


def loadVocab(dir_to_read):
    file_path = os.path.join(dir_to_read, vocab_file_name)
    with open(file_path, 'rb') as f:
        return cPickle.load(f)


def persistData(data, dir_to_save):
    file_path = os.path.join(dir_to_save, data_file_name)
    with open(file_path, 'wb') as f:
        cPickle.dump(data, f)


def loadData(dir_to_read):
    file_path = os.path.join(dir_to_read, data_file_name)
    with open(file_path, 'rb') as f:
        return cPickle.load(f)
