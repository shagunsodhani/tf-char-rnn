from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq
import numpy as np


class Model():
    def __init__(self, args):
        self.args = args
        model_to_cell_map = {
            'rnn': rnn_cell.BasicRNNCell,
            'gru': rnn_cell.GRUCell,
            'lstm': rnn_cell.BasicLSTMCell
        }
        self.device = args.device
        if not args.is_training:
            args.batch_size = 1
            args.seq_length = 1
        self.batch_size = args.batch_size
        self.seq_length = args.seq_length
        _cell = model_to_cell_map.get(args.model, rnn_cell.BasicRNNCell)(args.rnn_size)
        cell = self.cell = rnn_cell.MultiRNNCell([_cell] * args.num_layers)

        self._input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self._targets = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self._initial_state = cell.zero_state(args.batch_size, tf.float32)

        with tf.variable_scope('RNN'):
            with tf.device(self.device):
                embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size])
                inputs = tf.split(1, args.seq_length, tf.nn.embedding_lookup(embedding, self.input_data))
                inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
                softmax_w = tf.get_variable("softmax_w", [args.rnn_size, args.vocab_size], dtype=tf.float32)
                softmax_b = tf.get_variable("softmax_b", [args.vocab_size], dtype=tf.float32)

                def predict_char(prev, _):
                    prev = tf.matmul(prev, softmax_w) + softmax_b
                    prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
                    return tf.nn.embedding_lookup(embedding, prev_symbol)

                loop_function = None if args.is_training else predict_char

                outputs, self._final_state = seq2seq.rnn_decoder(inputs, self._initial_state, cell,
                                                                 loop_function=loop_function)
                output = tf.reshape(tf.concat(1, outputs), [-1, args.rnn_size])
                self.logits = logits = tf.matmul(output, softmax_w) + softmax_b
                self.probability = tf.nn.softmax(logits)
                loss = seq2seq. \
                    sequence_loss_by_example([logits], [tf.reshape(self._targets, [-1])],
                                             [tf.ones([args.batch_size * args.seq_length])],
                                             args.vocab_size)
                self._cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length

                if not args.is_training:
                    return

                self._lr = tf.Variable(0.0, trainable=False)
                tvars = tf.trainable_variables()
                grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),
                                                  args.grad_clip)
                optimizer = tf.train.AdamOptimizer(self._lr)
                # optimizer = tf.train.GradientDescentOptimizer(self._lr)
                self._train_op = optimizer.apply_gradients(zip(grads, tvars))

                self._new_lr = tf.placeholder(
                    tf.float32, shape=[], name="new_learning_rate")
                self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

    def sample(self, sess, chars, vocab, num=200, prime='The ', sampling_type=1):
        initial_state = self.cell.zero_state(self.args.batch_size, tf.float32).eval()
        for char in prime[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state:initial_state}
            [initial_state] = sess.run([self.final_state], feed)

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        to_return = prime
        prev_char = prime[-1]
        for n in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[prev_char]
            feed = {self.input_data: x, self.initial_state:initial_state}
            [probability, state] = sess.run([self.probability, self.final_state], feed)
            p = probability[0]

            if sampling_type == 0:
                sample = np.argmax(p)
            elif sampling_type == 2:
                if char == ' ':
                    sample = weighted_pick(p)
                else:
                    sample = np.argmax(p)
            else: # sampling_type == 1 default:
                sample = weighted_pick(p)

            predicted_char = chars[sample]
            to_return += predicted_char
            prev_char = predicted_char
        return to_return