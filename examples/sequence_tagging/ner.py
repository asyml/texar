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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import importlib
import numpy as np
import tensorflow as tf
import texar as tx

from conll_reader import create_vocabs, read_data, iterate_batch

flags = tf.flags

flags.DEFINE_string("data_path", "./data",
                    "Directory containing NER data (e.g., eng.train.bio.conll).")
flags.DEFINE_string("train", "eng.train.bio.conll",
                    "the file name of the training data.")
flags.DEFINE_string("dev", "eng.dev.bio.conll",
                    "the file name of the dev data.")
flags.DEFINE_string("test", "eng.test.bio.conll",
                    "the file name of the test data.")
flags.DEFINE_string("embedding", "glove.6B.100d.gz",
                    "the file name of the GloVe embedding.")
flags.DEFINE_string("config", "config", "The config to use.")

FLAGS = flags.FLAGS

config = importlib.import_module(FLAGS.config)

train_path = FLAGS.train
dev_path = FLAGS.dev
test_path = FLAGS.test
embedding_path = FLAGS.embedding
EMBEDD_DIM = 100

(word_vocab, char_vocab, ner_vocab), (i2w, i2n) = create_vocabs(train_path)

scale = np.sqrt(3.0 / EMBEDD_DIM)
word_vecs = np.random.uniform(-scale, scale, [len(word_vocab), EMBEDD_DIM]).astype(np.float32)

word_vecs = tx.data.load_glove(embedding_path, word_vocab, word_vecs)

data_train = read_data(train_path, word_vocab, char_vocab, ner_vocab)
data_dev = read_data(dev_path, word_vocab, char_vocab, ner_vocab)
data_test = read_data(test_path, word_vocab, char_vocab, ner_vocab)

vocab_size = len(word_vocab)

inputs = tf.placeholder(tf.int32, [None, None])
targets = tf.placeholder(tf.int32, [None, None])
masks = tf.placeholder(tf.float32, [None, None])
seq_lengths = tf.placeholder(tf.int32, [None])

embedder = tx.modules.WordEmbedder(vocab_size=vocab_size, init_value=word_vecs)
emb_inputs = embedder(inputs)

encoder = tx.modules.BidirectionalRNNEncoder(hparams={"rnn_cell_fw": config.cell, "rnn_cell_bw": config.cell})

outputs, _ = encoder(inputs, sequence_lengths=seq_lengths)

rnn_shape = outputs.shape

outputs = tf.reshape(outputs, (-1, outputs.shape[2]))

outputs = tf.layers.dense(outputs, config.tag_space, activation=tf.nn.elu)

logits = tf.layers.dense(outputs, len(ner_vocab))

logits = tf.reshape(logits, (rnn_shape[0], rnn_shape[1], len(ner_vocab)))

mle_loss = tx.losses.sequence_sparse_softmax_cross_entropy(
    labels=targets,
    logits=logits,
    sequence_length=seq_lengths,
    average_across_batch=True,
    average_across_timesteps=True)

predicts = tf.arg_max(logits, dimension=2)
corrects = tf.reduce_sum(tf.equal(targets, predicts) * masks)

global_step = tf.placeholder(tf.int32)
train_op = tx.core.get_train_op(
    mle_loss, global_step=global_step, increment_global_step=False,
    hparams=config.opt)

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
    for batch in iterate_batch(data_train, config.batch_size, shuffle=True):
        word, char, ner, mask, length = batch
        feed_dict = {
            inputs: word, targets: ner, masks: mask, seq_lengths: length,
            global_step: epoch, tx.global_mode(): mode,
        }

        rets = sess.run(fetches, feed_dict)
        nums = np.sum(length)
        loss += rets["mle_loss"] * nums
        corr += rets["correct"]
        num_tokens += nums

        print("train: %d loss: %.4f, acc: %.2f%%" % (epoch, loss / num_tokens, corr / num_tokens * 100))
    print("train: %d loss: %.4f, acc: %.2f%%, time: %.2fs" % (epoch, loss / num_tokens, corr / num_tokens * 100, time.time() - start_time))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(tf.tables_initializer())

    for epoch in range(config.num_epochs):
        _train_epoch(sess, epoch)



