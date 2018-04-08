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
"""Example for building the PTB language model.

This is a reimpmentation of the TensorFlow official PTB example in:
tensorflow/models/rnn/ptb

Model and training are described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
 http://arxiv.org/abs/1409.2329

The exact results may vary depending on the random initialization.

The data required for this example is in the `data/` dir of the
PTB dataset from Tomas Mikolov's webpage:

$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz

If data is now provided, the program will download from above automatically.

To run:

$ python lm_ptb.py --data_path=simple-examples/data --config=config_small
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

num_epochs = 30
hidden_size = 256
keep_prob = 0.5
batch_size = 30
emb_size = 256
new_kl_rate = 0

annealing_rate = 1.0 / 7000

latent_dims = 64

cell_hparams = {
    "type": "LSTMBlockCell",
    "kwargs": {
        "num_units": hidden_size,
        "forget_bias": 0.
    },
    "dropout": {"output_keep_prob": keep_prob},
    "num_layers": 1
}

emb_hparams = {
    "dim": emb_size
}

connector_hparams = {

}

train_data_hparams = {
    "num_epochs": 1,
    "batch_size": batch_size,
    "seed": 123,
    "dataset": {
        "files": '/space/hzt/text/data/ptb/ptb.train.txt',
        "vocab_file": '/space/hzt/text/data/ptb/vocab.txt'
    }
}

val_data_hparams = {
    "num_epochs": 1,
    "batch_size": batch_size,
    "seed": 123,
    "dataset": {
        "files": '/space/hzt/text/data/ptb/ptb.val.txt',
        "vocab_file": '/space/hzt/text/data/ptb/vocab.txt'
    }
}

test_data_hparams = {
    "num_epochs": 1,
    "batch_size": batch_size,
    "dataset": {
        "files": '/space/hzt/text/data/ptb/ptb.test.txt', # TODO(zhiting): use new data
        "vocab_file": '/space/hzt/text/data/ptb/vocab.txt'
    }
}

opt_hparams = {
    "optimizer": {
        "type": "GradientDescentOptimizer",
        "kwargs": {"learning_rate": 1.0}
    },
    "gradient_clip": {
        "type": "clip_by_global_norm",
        "kwargs": {"clip_norm": 5.}
    },
    "learning_rate_decay": {
        "type": "exponential_decay",
        "kwargs": {
            "decay_steps": 1,
            "decay_rate": 0.5,
            "staircase": True
        },
        "start_decay_step": 3
    }
}

# opt_hparams = {
#     "optimizer": {
#         "type": "AdamOptimizer",
#         "kwargs": {"learning_rate": 0.01}
#     },

#     "gradient_clip": {
#         "type": "clip_by_global_norm",
#         "kwargs": {"clip_norm": 5.}
#     },
# }

def kl_dvg(means, var):
    kl_cost = -0.5 * (tf.log(var) - tf.square(means) -
        var + 1.0)
    kl_cost = tf.reduce_mean(kl_cost, 0)

    return tf.reduce_sum(kl_cost)

def _main(_):
    # Data
    train_data = tx.data.MonoTextData(train_data_hparams)
    # val_data = tx.data.MonoTextData(val_data_hparams)
    test_data = tx.data.MonoTextData(test_data_hparams)
    iterator = tx.data.TrainTestDataIterator(train=train_data,
                                             test=test_data)
    data_batch = iterator.get_next()
    # data = prepare_data(FLAGS.data_path)
    # vocab_size = data["vocab_size"]

    # inputs = tf.placeholder(tf.int32, [batch_size, num_steps])
    # outputs = tf.placeholder(tf.int32, [batch_size, num_steps])
    # targets = tf.placeholder(tf.int32, [batch_size, num_steps])

    # Model architecture

    embedder = tx.modules.WordEmbedder(
        vocab_size=train_data.vocab.size, hparams=emb_hparams)

    # if config.keep_prob < 1:
    #     emb_inputs = tf.nn.dropout(
    #         emb_inputs, tx.utils.switch_dropout(config.keep_prob))
    #     emb_outputs = tf.nn.dropout(
    #         emb_outputs, tx.utils.switch_dropout(config.keep_prob))

    encoder = tx.modules.UnidirectionalRNNEncoder(
        hparams={"rnn_cell": cell_hparams})
    input_embedding = embedder(data_batch["text_ids"])
    if keep_prob < 1:
        input_embedding = tf.nn.dropout(
            input_embedding, tx.utils.switch_dropout(keep_prob))

    decoder = tx.modules.BasicRNNDecoder(
        vocab_size=train_data.vocab.size, hparams={"rnn_cell": cell_hparams})

    connector_mlp = tx.modules.connectors.MLPTransformConnector(
            latent_dims * 2, hparams={"activation_fn": "relu"})
    connector_stoch = tx.modules.connectors.ReparameterizedStochasticConnector(
            decoder.cell.state_size, hparams={"activation_fn": "relu"})

    # initial_state = decoder.zero_state(batch_size, tf.float32)
    _, ecdr_states = encoder(
        input_embedding,
        sequence_length=data_batch["length"])

    mean_logvar = connector_mlp(ecdr_states)
    mean, logvar = tf.split(mean_logvar, 2, 1)
    var = tf.nn.softplus(logvar)

    dst = tf.contrib.distributions.MultivariateNormalDiag(
        loc=mean,
        scale_diag=tf.sqrt(var))

    dcdr_states = connector_stoch(dst)

    outputs, final_state, seq_lengths = decoder(
        initial_state=dcdr_states,
        decoding_strategy="train_greedy",
        inputs=data_batch["text_ids"],
        sequence_length=data_batch["length"]-1,
        embedding=embedder)

    # Losses & train ops
    mle_loss = tx.losses.sequence_sparse_softmax_cross_entropy(
        labels=data_batch["text_ids"][:, 1:],
        logits=outputs.logits,
        sequence_length=seq_lengths)

    kl_loss = kl_dvg(mean, var)

    # annealing
    kl_rate = tf.placeholder(tf.float32, shape=())

    total_loss = mle_loss + kl_rate * kl_loss

    global_step = tf.placeholder(tf.int32)
    train_op = tx.core.get_train_op(total_loss,
                                    global_step=global_step,
                                    increment_global_step=False,
                                    hparams=opt_hparams)

    def _train_epochs(sess, epoch, display=10):
        global new_kl_rate

        start_time = time.time()
        iterator.switch_to_train_data(sess)
        num_words = 0
        t_nll = []
        t_kl_loss = []
        t_total_loss = []
        step = 0
        while True:
            try:
                fetches = {"train_op": train_op,
                           "total_loss": total_loss,
                           "mle_loss": mle_loss,
                           "kl_loss": kl_loss,
                           "lengths": seq_lengths}

                assert new_kl_rate <= 1.0
                feed = {tx.global_mode(): tf.estimator.ModeKeys.TRAIN,
                        kl_rate: new_kl_rate,
                        global_step: epoch}
                fetches_ = sess.run(fetches, feed_dict=feed)
                new_kl_rate = new_kl_rate + annealing_rate \
                              if new_kl_rate <= 1.0 - annealing_rate \
                              else 1.0

                num_words += sum(fetches_["lengths"])
                t_nll.append(fetches_["mle_loss"] + fetches_["kl_loss"])
                t_kl_loss.append(fetches_["kl_loss"])
                t_total_loss.append(fetches_["total_loss"])
                log_ppl = batch_size * np.sum(t_nll) / num_words

                if step % display == 0:
                    print('epoch %d, step %d, total_loss %.4f, \
                           nll %.4f, KL %.4f, log_pll %.4f, time %.1f' %
                          (epoch, step, np.mean(t_total_loss),
                           np.mean(t_nll), np.mean(t_kl_loss), log_ppl,
                           time.time() - start_time))

                step += 1

            except tf.errors.OutOfRangeError:
                print('\nepoch %d End, total_loss %.4f, \
                       nll %.4f, KL %.4f, log_pll %.4f' %
                      (epoch, np.mean(t_total_loss),
                       np.mean(t_nll), np.mean(t_kl_loss), log_ppl))
                break

    def _test_epochs(sess, epoch):
        iterator.switch_to_test_data(sess)
        num_words = 0
        t_nll = []
        t_kl_loss = []
        t_total_loss = []
        while True:
            try:
                fetches = {"total_loss": total_loss,
                           "mle_loss": mle_loss,
                           "kl_loss": kl_loss,
                           "lengths": seq_lengths}
                feed = {tx.global_mode(): tf.estimator.ModeKeys.TRAIN,
                        kl_rate: 1.0,
                        global_step: epoch}
                fetches_ = sess.run(fetches, feed_dict=feed)

                num_words += sum(fetches_["lengths"])
                t_nll.append(fetches_["mle_loss"] + fetches_["kl_loss"])
                t_kl_loss.append(fetches_["kl_loss"])
                t_total_loss.append(fetches_["total_loss"])


            except tf.errors.OutOfRangeError:
                log_ppl = batch_size * np.sum(t_nll) / num_words
                print('TEST: epoch %d, total_loss %.4f, \
                       nll %.4f, KL %.4f, pll %.4f\n' %
                      (epoch, np.mean(t_total_loss),
                       np.mean(t_nll), np.mean(t_kl_loss), log_ppl))
                break

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        for epoch in range(num_epochs):
            _train_epochs(sess, epoch, display=200)
            print('\nepoch %d, new kl rate %.4f\n' % (epoch, new_kl_rate))
            _test_epochs(sess, epoch)

if __name__ == '__main__':
    tf.app.run(main=_main)
