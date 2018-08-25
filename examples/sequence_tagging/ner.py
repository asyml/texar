# Copyright 2018 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Sequence tagging.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import importlib
import numpy as np
import tensorflow as tf
import texar as tx

from examples.sequence_tagging.conll_reader import create_vocabs, read_data, iterate_batch, load_glove, construct_init_word_vecs
from examples.sequence_tagging.conll_writer import CoNLLWriter
from examples.sequence_tagging import scores

flags = tf.flags

flags.DEFINE_string("data_path", "./data",
                    "Directory containing NER data (e.g., eng.train.bio.conll).")
flags.DEFINE_string("train", "eng.train.bio.conll",
                    "the file name of the training data.")
flags.DEFINE_string("dev", "eng.dev.bio.conll",
                    "the file name of the dev data.")
flags.DEFINE_string("test", "eng.test.bio.conll",
                    "the file name of the test data.")
flags.DEFINE_string("embedding", "glove.6B.100d.txt",
                    "the file name of the GloVe embedding.")
flags.DEFINE_string("config", "config", "The config to use.")

FLAGS = flags.FLAGS

config = importlib.import_module(FLAGS.config)

train_path = os.path.join(FLAGS.data_path, FLAGS.train)
dev_path = os.path.join(FLAGS.data_path, FLAGS.dev)
test_path = os.path.join(FLAGS.data_path, FLAGS.test)
embedding_path = os.path.join(FLAGS.data_path, FLAGS.embedding)
EMBEDD_DIM = config.embed_dim
CHAR_DIM = config.char_dim

# Prepares/loads data
if config.load_glove:
    print('loading GloVe embedding...')
    glove_dict = load_glove(embedding_path, EMBEDD_DIM)
else:
    glove_dict = None

(word_vocab, char_vocab, ner_vocab), (i2w, i2n) = create_vocabs(train_path, dev_path, test_path, glove_dict=glove_dict)

data_train = read_data(train_path, word_vocab, char_vocab, ner_vocab)
data_dev = read_data(dev_path, word_vocab, char_vocab, ner_vocab)
data_test = read_data(test_path, word_vocab, char_vocab, ner_vocab)

scale = np.sqrt(3.0 / EMBEDD_DIM)
word_vecs = np.random.uniform(-scale, scale, [len(word_vocab), EMBEDD_DIM]).astype(np.float32)
if config.load_glove:
    word_vecs = construct_init_word_vecs(word_vocab, word_vecs, glove_dict)

scale = np.sqrt(3.0 / CHAR_DIM)
char_vecs = np.random.uniform(-scale, scale, [len(char_vocab), CHAR_DIM]).astype(np.float32)

# Builds TF graph
inputs = tf.placeholder(tf.int64, [None, None])
chars = tf.placeholder(tf.int64, [None, None, None])
targets = tf.placeholder(tf.int64, [None, None])
masks = tf.placeholder(tf.float32, [None, None])
seq_lengths = tf.placeholder(tf.int64, [None])

vocab_size = len(word_vecs)
embedder = tx.modules.WordEmbedder(vocab_size=vocab_size, init_value=word_vecs, hparams=config.emb)
emb_inputs = embedder(inputs)

char_size = len(char_vecs)
char_embedder = tx.modules.WordEmbedder(vocab_size=char_size, init_value=char_vecs, hparams=config.char_emb)
emb_chars = char_embedder(chars)
char_shape = tf.shape(emb_chars) # [batch, length, char_length, char_dim]
emb_chars = tf.reshape(emb_chars, (-1, char_shape[2], CHAR_DIM))
char_encoder = tx.modules.Conv1DEncoder(config.conv)
char_outputs = char_encoder(emb_chars)
char_outputs = tf.reshape(char_outputs, (char_shape[0], char_shape[1], config.conv['filters']))

emb_inputs = tf.concat([emb_inputs, char_outputs], axis=2)
emb_inputs = tf.nn.dropout(emb_inputs, keep_prob=0.67)

encoder = tx.modules.BidirectionalRNNEncoder(hparams={"rnn_cell_fw": config.cell, "rnn_cell_bw": config.cell})
outputs, _ = encoder(emb_inputs, sequence_length=seq_lengths)
outputs = tf.concat(outputs, axis=2)

rnn_shape = tf.shape(outputs)
outputs = tf.reshape(outputs, (-1, 2 * config.hidden_size))

outputs = tf.layers.dense(outputs, config.tag_space, activation=tf.nn.elu)
outputs = tf.nn.dropout(outputs, keep_prob=config.keep_prob)

logits = tf.layers.dense(outputs, len(ner_vocab))

logits = tf.reshape(logits, tf.concat([rnn_shape[0:2], [len(ner_vocab)]], axis=0))

mle_loss = tx.losses.sequence_sparse_softmax_cross_entropy(
    labels=targets,
    logits=logits,
    sequence_length=seq_lengths,
    average_across_batch=True,
    average_across_timesteps=True,
    sum_over_timesteps=False)

predicts = tf.argmax(logits, axis=2)
corrects = tf.reduce_sum(tf.cast(tf.equal(targets, predicts), tf.float32) * masks)

global_step = tf.placeholder(tf.int32)
train_op = tx.core.get_train_op(
    mle_loss, global_step=global_step, increment_global_step=False,
    hparams=config.opt)

# Training/eval processes

def _train_epoch(sess, epoch):
    start_time = time.time()
    loss = 0.
    corr = 0.
    num_tokens = 0.

    fetches = {
        "mle_loss": mle_loss,
        "correct": corrects,
    }
    fetches["train_op"] = train_op

    mode = tf.estimator.ModeKeys.TRAIN
    num_inst = 0
    for batch in iterate_batch(data_train, config.batch_size, shuffle=True):
        word, char, ner, mask, length = batch
        feed_dict = {
            inputs: word, chars: char, targets: ner, masks: mask, seq_lengths: length,
            global_step: epoch, tx.global_mode(): mode,
        }

        rets = sess.run(fetches, feed_dict)
        nums = np.sum(length)
        num_inst += len(word)
        loss += rets["mle_loss"] * nums
        corr += rets["correct"]
        num_tokens += nums

        print("train: %d (%d/%d) loss: %.4f, acc: %.2f%%" % (epoch, num_inst, len(data_train), loss / num_tokens, corr / num_tokens * 100))
    print("train: %d loss: %.4f, acc: %.2f%%, time: %.2fs" % (epoch, loss / num_tokens, corr / num_tokens * 100, time.time() - start_time))


def _eval(sess, epoch, data_tag):
    fetches = {
        "predicts": predicts,
    }
    mode = tf.estimator.ModeKeys.EVAL
    file_name = 'tmp/%s%d' % (data_tag, epoch)
    writer = CoNLLWriter(i2w, i2n)
    writer.start(file_name)
    data = data_dev if data_tag == 'dev' else data_test
    for batch in iterate_batch(data, config.batch_size, shuffle=False):
        word, char, ner, mask, length = batch
        feed_dict = {
            inputs: word, chars: char, targets: ner, masks: mask, seq_lengths: length,
            global_step: epoch, tx.global_mode(): mode,
        }
        rets = sess.run(fetches, feed_dict)
        predictions = rets['predicts']
        writer.write(word, predictions, ner, length)
    writer.close()
    acc, precision, recall, f1 = scores.scores(file_name)
    print('%s acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%%' % (data_tag, acc, precision, recall, f1))
    return acc, precision, recall, f1


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(tf.tables_initializer())

    dev_f1 = 0.0
    dev_acc = 0.0
    dev_precision = 0.0
    dev_recall = 0.0
    best_epoch = 0

    test_f1 = 0.0
    test_acc = 0.0
    test_prec = 0.0
    test_recall = 0.0

    tx.utils.maybe_create_dir('./tmp')

    for epoch in range(config.num_epochs):
        _train_epoch(sess, epoch)
        acc, precision, recall, f1 = _eval(sess, epoch, 'dev')
        if dev_f1 < f1:
            dev_f1 = f1
            dev_acc = acc
            dev_precision = precision
            dev_recall = recall
            best_epoch = epoch
            test_acc, test_prec, test_recall, test_f1 = _eval(sess, epoch, 'test')
        print('best acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%%, epoch: %d' % (dev_acc, dev_precision, dev_recall, dev_f1, best_epoch))
        print('test acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%%, epoch: %d' % (test_acc, test_prec, test_recall, test_f1, best_epoch))
        print('---------------------------------------------------')
