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

import sys
import time
import importlib
import numpy as np
import tensorflow as tf
import texar as tx

flags = tf.flags

flags.DEFINE_string("config", "config", "The config to use.")

FLAGS = flags.FLAGS

config = importlib.import_module(FLAGS.config)

def kl_dvg(means, logvars):
    """compute the KL divergence between Gaussian distribution
    """
    kl_cost = -0.5 * (logvars - tf.square(means) -
                      tf.exp(logvars) + 1.0)
    kl_cost = tf.reduce_mean(kl_cost, 0)

    return tf.reduce_sum(kl_cost)


def _main(_):

    # Data
    train_data = tx.data.MonoTextData(config.train_data_hparams)
    val_data = tx.data.MonoTextData(config.val_data_hparams)
    test_data = tx.data.MonoTextData(config.test_data_hparams)
    iterator = tx.data.TrainTestDataIterator(train=train_data,
                                             val=val_data,
                                             test=test_data)
    data_batch = iterator.get_next()

    opt_vars = {
        'learning_rate': config.lr_decay_hparams["init_lr"],
        'best_valid_nll': 1e100,
        'steps_not_improved': 0,
        'kl_weight': config.kl_anneal_hparams["start"]
    }

    # KL term annealing rate
    anneal_r = 1.0 / (config.kl_anneal_hparams["warm_up"] * \
        (train_data.dataset_size() / config.batch_size))

    # Model architecture
    embedder = tx.modules.WordEmbedder(
        vocab_size=train_data.vocab.size, hparams=config.emb_hparams)


    output_embed = input_embed = embedder(data_batch["text_ids"])

    if config.enc_keep_prob_in < 1:
        input_embed = tf.nn.dropout(
            input_embed, tx.utils.switch_dropout(config.enc_keep_prob_in))

    if config.dec_keep_prob_in < 1:
        output_embed = tf.nn.dropout(
            output_embed, tx.utils.switch_dropout(config.dec_keep_prob_in))

    encoder = tx.modules.UnidirectionalRNNEncoder(
        hparams={"rnn_cell": config.enc_cell_hparams})

    if config.decoder_hparams["type"] == "lstm":
        decoder = tx.modules.BasicRNNDecoder(vocab_size=train_data.vocab.size,
            hparams={"rnn_cell": config.dec_cell_hparams})
        decoder_initial_state_size = decoder.cell.state_size
    elif config.decoder_hparams["type"] == 'transformer':
        decoder = tx.modules.TransformerDecoder(
            embedding=embedder._embedding,
            hparams=config.trans_hparams)
        decoder_initial_state_size = tf.TensorShape([1,
            config.emb_hparams["dim"]])
    else:
        raise NotImplementedError

    connector_mlp = tx.modules.connectors.MLPTransformConnector(
        config.latent_dims * 2)

    connector_stoch = tx.modules.connectors.ReparameterizedStochasticConnector(
        decoder_initial_state_size)

    _, ecdr_states = encoder(
        input_embed,
        sequence_length=data_batch["length"])

    mean_logvar = connector_mlp(ecdr_states)
    mean, logvar = tf.split(mean_logvar, 2, 1)
    kl_loss = kl_dvg(mean, logvar)

    dst = tf.contrib.distributions.MultivariateNormalDiag(
        loc=mean,
        scale_diag=tf.exp(0.5 * logvar))

    dcdr_states, _ = connector_stoch(dst)

    # decoder
    if config.decoder_hparams["type"] == "lstm":
        outputs, _, _ = decoder(
            initial_state=dcdr_states,
            decoding_strategy="train_greedy",
            inputs=output_embed,
            sequence_length=data_batch["length"]-1)
        logits = outputs.logits
    else:
        logits, _ = decoder(
            decoder_input=data_batch["text_ids"][:, :-1],
            encoder_output=dcdr_states,
            encoder_decoder_attention_bias=None)

    seq_lengths = data_batch["length"]-1

    # Losses & train ops
    rc_loss = tx.losses.sequence_sparse_softmax_cross_entropy(
        labels=data_batch["text_ids"][:, 1:],
        logits=logits,
        sequence_length=data_batch["length"]-1)

    # annealing
    kl_weight = tf.placeholder(tf.float32, shape=())

    nll = rc_loss + kl_weight * kl_loss

    # global_step = tf.placeholder(tf.int32)
    # train_op = tx.core.get_train_op(nll,
    #                                 global_step=global_step,
    #                                 increment_global_step=False,
    #                                 hparams=config.opt_hparams)

    global_step = tf.Variable(0, dtype=tf.int32)
    learning_rate = \
        tf.placeholder(dtype=tf.float32, shape=(), name='learning_rate')
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        beta1=0.,
        beta2=0.999,
        epsilon=1e-9)

    gradients, variables = zip(*optimizer.compute_gradients(nll))
    gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    train_op = optimizer.apply_gradients(zip(gradients, variables),
                                         global_step=global_step)
    # train_op = optimizer.minimize(
    #     nll, global_step=global_step)

    def _train_epochs(sess, epoch, step, display=10):
        start_time = time.time()
        iterator.switch_to_train_data(sess)
        num_words = num_sents = 0
        nll_ = 0.
        kl_loss_ = rc_loss_ = 0.
        while True:
            try:
                fetches = {"train_op": train_op,
                           "nll": nll,
                           "kl_loss": kl_loss,
                           "rc_loss": rc_loss,
                           "lengths": seq_lengths}

                opt_vars["kl_weight"] = min(1.0,
                                            opt_vars["kl_weight"] + anneal_r)
                feed = {tx.global_mode(): tf.estimator.ModeKeys.TRAIN,
                        kl_weight: opt_vars["kl_weight"],
                        learning_rate: opt_vars["learning_rate"]}

                 #fetch_debug = {"debug_data": debug_data,
                 #                "logits": logits,
                 #                "rc_loss": rc_loss}
                 #fetch_debug_ = sess.run(fetch_debug, feed_dict=feed)

                fetches_ = sess.run(fetches, feed_dict=feed)

                batch_size = len(fetches_["lengths"])
                num_sents += batch_size

                num_words += sum(fetches_["lengths"])
                nll_ += fetches_["nll"] * batch_size
                kl_loss_ += fetches_["kl_loss"] * batch_size
                rc_loss_ += fetches_["rc_loss"] * batch_size

                if step % display == 0:
                    print('epoch %d, step %d, nll %.4f, klw: %.4f, KL %.4f,  ' \
                           'rc %.4f, log_ppl %.4f, ppl %.4f, ' \
                           'time elapsed: %.1fs' % \
                          (epoch, step, nll_ / num_sents, opt_vars["kl_weight"],
                           kl_loss_ / num_sents, rc_loss_ / num_sents,
                           nll_ / num_words, np.exp(nll_ / num_words),
                           time.time() - start_time))

                    sys.stdout.flush()

                step += 1

            except tf.errors.OutOfRangeError:
                break
        return

    def _val_epochs(sess, epoch):
        iterator.switch_to_val_data(sess)
        num_words = num_sents = 0
        nll_ = 0.
        kl_loss_ = rc_loss_ = 0.
        while True:
            try:
                fetches = {"nll": nll,
                           "kl_loss": kl_loss,
                           "rc_loss": rc_loss,
                           "lengths": seq_lengths}
                feed = {tx.global_mode(): tf.estimator.ModeKeys.EVAL,
                        kl_weight: 1.0,
                        learning_rate: opt_vars["learning_rate"]}
                fetches_ = sess.run(fetches, feed_dict=feed)

                batch_size = len(fetches_["lengths"])

                num_sents += batch_size

                num_words += sum(fetches_["lengths"])
                nll_ += fetches_["nll"] * batch_size
                kl_loss_ += fetches_["kl_loss"] * batch_size
                rc_loss_ += fetches_["rc_loss"] * batch_size

            except tf.errors.OutOfRangeError:
                print('\nVAL: epoch %d, nll %.4f, KL %.4f, rc %.4f, ' \
                      'log_ppl %.4f, ppl %.4f\n' %
                      (epoch, nll_ / num_sents, kl_loss_ / num_sents,
                       rc_loss_ / num_sents, nll_ / num_words,
                       np.exp(nll_ / num_words)))
                break

        return nll_ / num_sents, np.exp(nll_ / num_words)

    def _test_epochs(sess, epoch):
        iterator.switch_to_test_data(sess)
        num_words = num_sents = 0
        nll_ = 0.
        kl_loss_ = rc_loss_ = 0.
        while True:
            try:
                fetches = {"nll": nll,
                           "kl_loss": kl_loss,
                           "rc_loss": rc_loss,
                           "lengths": seq_lengths}
                feed = {tx.global_mode(): tf.estimator.ModeKeys.EVAL,
                        kl_weight: 1.0,
                        learning_rate: opt_vars["learning_rate"]}
                fetches_ = sess.run(fetches, feed_dict=feed)

                batch_size = len(fetches_["lengths"])

                num_sents += batch_size

                num_words += sum(fetches_["lengths"])
                nll_ += fetches_["nll"] * batch_size
                kl_loss_ += fetches_["kl_loss"] * batch_size
                rc_loss_ += fetches_["rc_loss"] * batch_size

            except tf.errors.OutOfRangeError:
                print('\nTEST: epoch %d, nll %.4f, KL %.4f, rc %.4f, ' \
                      'log_ppl %.4f, ppl %.4f\n' %
                      (epoch, nll_ / num_sents, kl_loss_ / num_sents,
                       rc_loss_ / num_sents, nll_ / num_words,
                       np.exp(nll_ / num_words)))
                break

        return nll_ / num_sents, np.exp(nll_ / num_words)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        # count trainable parameters
        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters

        print("%d total parameters" % total_parameters)

        step = 0
        best_nll = best_ppl = 0.

        for epoch in range(config.num_epochs):
            _train_epochs(sess, epoch, step, display=200)
            val_nll, val_ppl = _val_epochs(sess, epoch)
            test_nll, test_ppl = _test_epochs(sess, epoch)

            if val_nll < opt_vars['best_valid_nll']:
                opt_vars['best_valid_nll'] = val_nll
                opt_vars['steps_not_improved'] = 0
                best_nll = test_nll
                best_ppl = test_ppl
            else:
                opt_vars['steps_not_improved'] += 1
                if opt_vars['steps_not_improved'] == \
                config.lr_decay_hparams["threshold"]:
                    old_lr = opt_vars['learning_rate']
                    opt_vars['learning_rate'] *= config.lr_decay_hparams["rate"]
                    opt_vars['steps_not_improved'] = 0
                    new_lr = opt_vars['learning_rate']
                    print('-----\nchange lr, old lr: %f, new lr: %f\n-----' %
                          (old_lr, new_lr))

        print('\nbest testing nll: %.4f, best testing ppl %.4f\n' %
              (best_nll, best_ppl))


if __name__ == '__main__':
    tf.app.run(main=_main)
