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
    sample(args)

def sample(args):
    with open(os.path.join(args.model_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
        saved_args.is_training = False
    with open(os.path.join(args.model_dir, 'chars_vocab.pkl'), 'rb') as f:
        chars, vocab = cPickle.load(f)

    with tf.Session(config=tf.ConfigProto(
                      allow_soft_placement=True)) as sess:
        with tf.variable_scope("model", reuse=None):
            model = Model(saved_args)
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(args.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print(model.sample(sess, chars, vocab, args.n, args.prime, args.sample))

if __name__ == '__main__':
    main()