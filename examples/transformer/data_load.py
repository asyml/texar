# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
from __future__ import print_function
from hyperparams import args as hp
import tensorflow as tf
import numpy as np
import codecs
import regex

def load_shared_vocab():
    vocab = [line.split()[0] for line in codecs.open(hp.vocab_dir + 'vocab.bpe.32000.eval').read().splitlines()]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def create_data(source_sents, target_sents):
    word2idx, idx2word = load_shared_vocab()

    # Index
    x_list, y_list, Sources, Targets = [], [], [], []
    for source_sent, target_sent in zip(source_sents, target_sents):
        x = [word2idx[word] for word in (source_sent + u" <EOS>").split()]
        y = [word2idx[word] for word in (target_sent + u" <EOS>").split()]
        x = x[:hp.maxlen]
        y = y[:hp.maxlen]

        # truncate or discard
        #if max(len(x), len(y)) <=hp.maxlen:
        x_list.append(np.array(x))
        y_list.append(np.array(y))
        Sources.append(source_sent)
        Targets.append(target_sent)
    # Pad
    X = np.zeros([len(x_list), hp.maxlen], np.int32)
    Y = np.zeros([len(y_list), hp.maxlen], np.int32)
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        X[i] = np.lib.pad(x, [0, hp.maxlen-len(x)], 'constant', constant_values=(0, 0))
        Y[i] = np.lib.pad(y, [0, hp.maxlen-len(y)], 'constant', constant_values=(0, 0))

    return X, Y, Sources, Targets

def load_train_data():
    de_sents = [regex.sub("[^\s\p{Latin}']", "", line) for line in codecs.open(hp.source_train, 'r', 'utf-8').read().split("\n") if line and line[0] != "<"]
    en_sents = [regex.sub("[^\s\p{Latin}']", "", line) for line in codecs.open(hp.target_train, 'r', 'utf-8').read().split("\n") if line and line[0] != "<"]

    X, Y, Sources, Targets = create_data(de_sents, en_sents)
    return X, Y

def load_test_data():
    src_sents = [line for line in codecs.open(hp.source_test, 'r', 'utf-8').read().split("\n") if line]
    tgt_sents = [line for line in codecs.open(hp.target_test, 'r', 'utf-8').read().split("\n") if line]

    #all_sents = list(zip(src_sents, tgt_sents))
    #all_sents = sorted(all_sents, key=lambda x:len(x[1]))
    #src_sents, tgt_sents = list(zip(*all_sents))
    X, Y, Sources, Targets = create_data(src_sents, tgt_sents)
    return X, Sources, Targets # (1064, 150)

def get_batch_data():
    # Load data
    X, Y = load_train_data()

    # calc total batch count
    num_batch = len(X) // hp.batch_size

    # Convert to tensor
    X = tf.convert_to_tensor(X, tf.int32)
    Y = tf.convert_to_tensor(Y, tf.int32)

    # Create Queues
    input_queues = tf.train.slice_input_producer([X, Y])
    #actually the input data has already been shuffled

    # create batch queues
    x, y = tf.train.shuffle_batch(input_queues,
                                num_threads=8,
                                batch_size=hp.batch_size,
                                capacity=hp.batch_size*64,
                                min_after_dequeue=hp.batch_size*32,
                                allow_smaller_final_batch=False)

    return x, y, num_batch # (N, T), (N, T), ()

