from __future__ import print_function

import os
import time

import numpy as np
import tensorflow as tf
import cPickle

from model.model import Model
from reader import filereader
from utils import argumentparser


def main():
    args = argumentparser.ArgumentParser()
    train(args)


def train(args):
    train_data, vocab, chars = filereader.read_raw_data(args.input_dir)
    filereader.persistVocab(vocab, args.model_dir)
    args.is_training = True
    args.vocab_size = len(vocab)
    with open(os.path.join(args.model_dir, 'config.pkl'), 'wb') as f:
        cPickle.dump(args, f)
    with open(os.path.join(args.model_dir, 'chars_vocab.pkl'), 'wb') as f:
        cPickle.dump((chars, vocab), f)
    with tf.Session(config=tf.ConfigProto(
                      allow_soft_placement=True)) as sess:
    # with tf.Session() as sess:
        with tf.variable_scope("model", reuse=None):
            model = Model(args)
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        if args.init_from is not None:
            saver.restore(sess, args.model_checkpoint_path)
        for epoch in xrange(args.num_epochs):
            lr_decay = args.decay_rate ** epoch
            # lr_decay = 1
            model.assign_lr(sess, args.learning_rate * lr_decay)
            model_checkpoint_path = os.path.join(args.model_dir, 'model-epoch-' + str(epoch) + '.ckpt')
            train_perplexity = run_epoch(sess, model, train_data, model.train_op, \
                                         saver, model_checkpoint_path, args.save_frequency,
                                         verbose=True)
            info_statement = "Epoch: %d Learning rate: %.6f Perplexity: %.3f" % \
                             (epoch, sess.run(model.lr), train_perplexity)

            print(info_statement)
        model_checkpoint_path = os.path.join(args.model_dir, 'model-epoch-' + str(epoch) + '-complete.ckpt')
        saver.save(sess, model_checkpoint_path)
        print("model saved to {}".format(model_checkpoint_path))


def run_epoch(sess, model, data, eval_op, saver, model_checkpoint_path, save_frequency,
              verbose=False):
    epoch_size = ((len(data) // model.batch_size) - 1) // model.seq_length
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = sess.run(model.initial_state)
    for step, (x, y) in enumerate(filereader.batch_iterator(data, model.batch_size,
                                                            model.seq_length)):
        fetches = [model.cost, model.final_state, eval_op]
        feed_dict = {}
        feed_dict[model.input_data] = x
        feed_dict[model.targets] = y
        feed_dict[model.initial_state] = state
        cost, state, _ = sess.run(fetches, feed_dict)
        costs += cost
        iters += model.args.seq_length

        if (step) % save_frequency == 0:
            saver.save(sess, model_checkpoint_path, global_step=step)
            print("model saved to {}".format(model_checkpoint_path))
            if verbose:
                print("%.3f perplexity: %.3f cost: %.3f time/batch: %.3f" %
                      (step * 1.0 / epoch_size, np.exp(costs / iters), cost,
                       (time.time() - start_time)))


    return np.exp(costs / iters)


if __name__ == '__main__':
    main()
