from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq


class Model():
    def __init__(self, args):
        self.args = args
        model_to_cell_map = {
            'rnn': rnn_cell.BasicRNNCell,
            'gru': rnn_cell.GRUCell,
            'lstm': rnn_cell.BasicLSTMCell
        }
        self.batch_size = args.batch_size
        self.seq_length = args.seq_length
        _cell = model_to_cell_map.get(args.model, rnn_cell.BasicRNNCell)(args.rnn_size)
        cell = self.cell = rnn_cell.MultiRNNCell([_cell] * args.num_layers)

        self._input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self._targets = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self._initial_state = cell.zero_state(args.batch_size, tf.float32)

        with tf.variable_scope('RNN'):
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
