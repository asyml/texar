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
"""Example for building the language model.

This is a reimpmentation of the TensorFlow official PTB example in:
tensorflow/models/rnn/ptb

Model and training are described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
 http://arxiv.org/abs/1409.2329

There are 3 provided model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

The data required for this example is in the `data/` dir of the
PTB dataset from Tomas Mikolov's webpage:

$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz

If data is not provided, the program will download from above automatically.

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

from ptb_reader import prepare_data, ptb_iterator

flags = tf.flags

flags.DEFINE_string("data_path", "./",
                    "Directory containing PTB raw data (e.g., ptb.train.txt). "
                    "E.g., ./simple-examples/data. If not exists, "
                    "the directory will be created and PTB raw data will "
                    "be downloaded.")
flags.DEFINE_string("config", "config_small", "The config to use.")

FLAGS = flags.FLAGS

config = importlib.import_module(FLAGS.config)

def _main(_):
    # Data
    batch_size = config.batch_size
    num_steps = config.num_steps
    data = prepare_data(FLAGS.data_path)
    vocab_size = data["vocab_size"]

    inputs = tf.placeholder(tf.int32, [batch_size, num_steps])
    targets = tf.placeholder(tf.int32, [batch_size, num_steps])

    # Model architecture
    initializer = tf.random_uniform_initializer(
        -config.init_scale, config.init_scale)
    with tf.variable_scope("model", initializer=initializer):
        embedder = tx.modules.WordEmbedder(
            vocab_size=vocab_size, hparams=config.emb)
        emb_inputs = embedder(inputs)
        if config.keep_prob < 1:
            emb_inputs = tf.nn.dropout(
                emb_inputs, tx.utils.switch_dropout(config.keep_prob))

        decoder = tx.modules.BasicRNNDecoder(
            vocab_size=vocab_size, hparams={"rnn_cell": config.cell})
        initial_state = decoder.zero_state(batch_size, tf.float32)
        outputs, final_state, seq_lengths = decoder(
            decoding_strategy="train_greedy",
            impute_finished=True,
            inputs=emb_inputs,
            sequence_length=[num_steps]*batch_size,
            initial_state=initial_state)

    # Losses & train ops
    mle_loss = tx.losses.sequence_sparse_softmax_cross_entropy(
        labels=targets,
        logits=outputs.logits,
        sequence_length=seq_lengths)

    # Use global_step to pass epoch, for lr decay
    global_step = tf.placeholder(tf.int32)
    train_op = tx.core.get_train_op(
        mle_loss, global_step=global_step, increment_global_step=False,
        hparams=config.opt)

    def _run_epoch(sess, data_iter, epoch, is_train=False, verbose=False):
        start_time = time.time()
        loss = 0.
        iters = 0
        state = sess.run(initial_state)

        fetches = {
            "mle_loss": mle_loss,
            "final_state": final_state,
        }
        if is_train:
            fetches["train_op"] = train_op

        mode = (tf.estimator.ModeKeys.TRAIN
                if is_train
                else tf.estimator.ModeKeys.EVAL)

        epoch_size = (len(data) // batch_size - 1) // num_steps
        for step, (x, y) in enumerate(data_iter):
            feed_dict = {
                inputs: x, targets: y, global_step: epoch,
                tx.global_mode(): mode,
            }
            for i, (c, h) in enumerate(initial_state):
                feed_dict[c] = state[i].c
                feed_dict[h] = state[i].h

            rets = sess.run(fetches, feed_dict)
            loss += rets["mle_loss"]
            state = rets["final_state"]
            iters += num_steps

            ppl = np.exp(loss / iters)
            if verbose and step % (epoch_size // 10) == 10:
                print("%.3f perplexity: %.3f speed: %.0f wps" %
                      (step * 1.0 / epoch_size, ppl,
                       iters * batch_size / (time.time() - start_time)))

        ppl = np.exp(loss / iters)
        return ppl

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        for epoch in range(config.num_epochs):
            # Train
            train_data_iter = ptb_iterator(
                data["train_text_id"], config.batch_size, num_steps)
            train_ppl = _run_epoch(
                sess, train_data_iter, epoch, is_train=True, verbose=True)
            print("Epoch: %d Train Perplexity: %.3f" % (epoch, train_ppl))
            # Valid
            valid_data_iter = ptb_iterator(
                data["valid_text_id"], config.batch_size, num_steps)
            valid_ppl = _run_epoch(sess, valid_data_iter, epoch)
            print("Epoch: %d Valid Perplexity: %.3f" % (epoch, valid_ppl))
        # Test
        test_data_iter = ptb_iterator(
            data["test_text_id"], batch_size, num_steps)
        test_ppl = _run_epoch(sess, test_data_iter, 0)
        print("Test Perplexity: %.3f" % (test_ppl))

if __name__ == '__main__':
    tf.app.run(main=_main)
