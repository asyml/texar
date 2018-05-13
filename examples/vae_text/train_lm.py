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
"""Example for building the Variational Autoencoder.

This is an impmentation of Variational Autoencoder for text generation

Model is described in:
(Bowman, et. al.) Generating Sentences from a Continuous Space
 https://arxiv.org/abs/1511.06349

To run:

$ python train.py

Hyperparameters and data path may be specified in config.py

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name, no-member, too-many-locals

import time
import importlib
import numpy as np
import tensorflow as tf
import texar as tx

from ptb_reader import prepare_data, ptb_iterator
from embedding_tied_language_model import EmbeddingTiedLanguageModel

flags = tf.flags

flags.DEFINE_string("data_path", "./",
                    "Directory containing PTB raw data (e.g., ptb.train.txt). "
                    "E.g., ./simple-examples/data. If not exists, "
                    "the directory will be created and PTB raw data will "
                    "be downloaded.")
flags.DEFINE_string("config", "config_small", "The config to use.")

FLAGS = flags.FLAGS

config = importlib.import_module(FLAGS.config)

def kl_dvg(means, logvars):
    kl_cost = -0.5 * (logvars - tf.square(means) -
                      tf.exp(logvars) + 1.0)
    kl_cost = tf.reduce_mean(kl_cost, 0)

    return tf.reduce_sum(kl_cost)

def kl_anneal_function(step, k, x0):
    return 1.0 / (1 + np.exp(-k*(step-x0)))


def _main(_):
    # Data
    train_data = tx.data.MonoTextData(config.train_data_hparams)
    val_data = tx.data.MonoTextData(config.val_data_hparams)
    test_data = tx.data.MonoTextData(config.test_data_hparams)
    iterator = tx.data.TrainTestDataIterator(train=train_data,
                                             val=val_data,
                                             test=test_data)
    data_batch = iterator.get_next()


    # Data
    batch_size = tf.placeholder(dtype=tf.int32, shape=())
    num_steps = config.num_steps
    data = prepare_data(FLAGS.data_path)
    vocab_size = data["vocab_size"]

    # Model architecture
    embedder = tx.modules.WordEmbedder(
        vocab_size=train_data.vocab.size, hparams=config.emb_hparams)


    encoder = tx.modules.UnidirectionalRNNEncoder(
        hparams={"rnn_cell": config.cell_hparams})

    output_embed = input_embed = embedder(data_batch["text_ids"])

    if config.keep_prob < 1:
        output_embed = tf.nn.dropout(
            output_embed, tx.utils.switch_dropout(config.keep_prob))


    decoder = EmbeddingTiedLanguageModel(vocab_size=vocab_size)

    connector_mlp = tx.modules.connectors.MLPTransformConnector(
            config.latent_dims * 2)
    connector_stoch = tx.modules.connectors.ReparameterizedStochasticConnector(
            decoder.cell.state_size)

    _, ecdr_states = encoder(
        input_embed,
        sequence_length=data_batch["length"])

    mean_logvar = connector_mlp(ecdr_states)
    mean, logvar = tf.split(mean_logvar, 2, 1)

    dst = tf.contrib.distributions.MultivariateNormalDiag(
        loc=mean,
        scale_diag=tf.exp(0.5 * logvar))

    dcdr_states = connector_stoch(dst)

    outputs, final_state, seq_lengths = decoder(
        initial_state=dcdr_states,
        decoding_strategy="train_greedy",
        inputs=output_embed,
        sequence_length=data_batch["length"]-1)

    # Losses & train ops
    rc_loss = tx.losses.sequence_sparse_softmax_cross_entropy(
        labels=data_batch["text_ids"][:, 1:],
        logits=outputs.logits,
        sequence_length=seq_lengths)

    kl_loss = kl_dvg(mean, logvar)

    # annealing
    kl_weight = tf.placeholder(tf.float32, shape=())

    nll = rc_loss + kl_weight * kl_loss

    global_step = tf.placeholder(tf.int32)
    train_op = tx.core.get_train_op(nll,
                                    global_step=global_step,
                                    increment_global_step=False,
                                    hparams=config.opt_hparams)

    def _train_epochs(sess, epoch, step, display=10):
        start_time = time.time()
        iterator.switch_to_train_data(sess)
        num_words = 0
        nll_ = []
        kl_loss_ = []
        while True:
            try:
                fetches = {"train_op": train_op,
                           "nll": nll,
                           "kl_loss": kl_loss,
                           "lengths": seq_lengths}

                kl_weight_n = kl_anneal_function(step,
                                                 config.anneal_hparams["k"],
                                                 config.anneal_hparams["x0"])
                feed = {tx.global_mode(): tf.estimator.ModeKeys.TRAIN,
                        kl_weight: kl_weight_n,
                        global_step: epoch}
                fetches_ = sess.run(fetches, feed_dict=feed)

                num_words += sum(fetches_["lengths"])
                nll_.append(fetches_["nll"])
                kl_loss_.append(fetches_["kl_loss"])
                log_ppl = config.batch_size * np.sum(nll_) / num_words

                if step % display == 0:
                    print('epoch %d, step %d, nll %.4f, KL %.4f, \
                           log_pll %.4f, time elapsed: %.1fs' % \
                          (epoch, step, np.mean(nll_), np.mean(kl_loss_),
                           log_ppl, time.time() - start_time))

                step += 1

            except tf.errors.OutOfRangeError:
                print('\nepoch %d finished, nll %.4f, \
                        KL %.4f, log_pll %.4f' %
                      (epoch, np.mean(nll_), np.mean(kl_loss_), log_ppl))
                break
        return step, kl_weight_n

    def _val_epochs(sess, epoch):
        iterator.switch_to_val_data(sess)
        num_words = 0
        nll_ = []
        kl_loss_ = []
        while True:
            try:
                fetches = {"nll": nll,
                           "kl_loss": kl_loss,
                           "lengths": seq_lengths}
                feed = {tx.global_mode(): tf.estimator.ModeKeys.EVAL,
                        kl_weight: 1.0,
                        global_step: epoch}
                fetches_ = sess.run(fetches, feed_dict=feed)

                num_words += sum(fetches_["lengths"])
                nll_.append(fetches_["nll"])
                kl_loss_.append(fetches_["kl_loss"])

            except tf.errors.OutOfRangeError:
                log_ppl = config.batch_size * np.sum(nll_) / num_words
                print('VAL: epoch %d, nll %.4f, KL %.4f, pll %.4f\n' %
                      (epoch, np.mean(nll_), np.mean(kl_loss_), log_ppl))
                break

    def _test_epochs(sess, epoch):
        iterator.switch_to_test_data(sess)
        num_words = 0
        nll_ = []
        kl_loss_ = []
        while True:
            try:
                fetches = {"nll": nll,
                           "kl_loss": kl_loss,
                           "lengths": seq_lengths}
                feed = {tx.global_mode(): tf.estimator.ModeKeys.EVAL,
                        kl_weight: 1.0,
                        global_step: epoch}
                fetches_ = sess.run(fetches, feed_dict=feed)

                num_words += sum(fetches_["lengths"])
                nll_.append(fetches_["nll"])
                kl_loss_.append(fetches_["kl_loss"])

            except tf.errors.OutOfRangeError:
                log_ppl = config.batch_size * np.sum(nll_) / num_words
                print('TEST: epoch %d, nll %.4f, KL %.4f, pll %.4f\n' %
                      (epoch, np.mean(nll_), np.mean(kl_loss_), log_ppl))
                break

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())
        step = 0

        for epoch in range(config.num_epochs):
            step, kl_weight_n = _train_epochs(sess, epoch, step, display=200)
            print('epoch %d, kl weight %.4f' % (epoch, kl_weight_n))
            _val_epochs(sess, epoch)
            _test_epochs(sess, epoch)

if __name__ == '__main__':
    tf.app.run(main=_main)
