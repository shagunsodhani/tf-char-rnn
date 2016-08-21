from __future__ import division
from __future__ import print_function

import collections
import os
import codecs

import numpy as np
import tensorflow as tf


def _read_file(file_path):
    with tf.gfile.GFile(file_path) as f:
        return f.read()


def _build_vocab(raw_data):
    counter = collections.Counter(raw_data)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    chars, _ = zip(*count_pairs)
    char_to_id = dict(zip(chars, range(len(chars))))
    return char_to_id


def _raw_data_to_char_ids(raw_data, char_to_id):
    return np.array([char_to_id[char] for char in raw_data], dtype=np.int32)


def read_raw_data(input_dir):
    train_path = os.path.join(input_dir, "data.txt")
    raw_data = _read_file(train_path)
    char_to_id = _build_vocab(raw_data)
    train_data = _raw_data_to_char_ids(raw_data, char_to_id)
    return train_data


def batch_iterator(train_data, batch_size, seq_length):
    train_data_len = len(train_data)
    num_batch = int(train_data_len // (batch_size * seq_length))
    if num_batch == 0:
        raise ValueError("num_batch == 0, decrease batch_size or seq_length")
    train_data = train_data[:num_batch * batch_size * seq_length]
    _x_train_data = train_data
    _y_train_data = np.copy(train_data)
    _y_train_data[:-1] = _x_train_data[1:]
    _y_train_data[-1] = _x_train_data[0]
    x_train_data = np.split(_x_train_data.reshape(batch_size, -1), num_batch, 1)
    y_train_data = np.split(_y_train_data.reshape(batch_size, -1), num_batch, 1)
    for i in xrange(len(x_train_data)):
        yield (x_train_data[i], y_train_data[i])