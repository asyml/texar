#!/usr/bin/env python3
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
"""Example for building the PTB language model with Memory Network.

Memory Network model is described in https://arxiv.org/abs/1503.08895v4

The data required for this example is in the `data/` dir of the
PTB dataset from Tomas Mikolov's webpage:

$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz

If data is now provided, the program will download from above automatically.

To run:

$ python lm_ptb_memnet.py --data_path=simple-examples/data \
  --config=config

This code will automatically save and restore from directory `ckpt/`.
If the directory doesn't exist, it will be created automatically.
"""

# pylint: disable=invalid-name, no-member, too-many-locals

import importlib
import numpy as np
import tensorflow as tf
import texar.tf as tx

from ptb_reader import prepare_data
from ptb_reader import ptb_iterator_memnet as ptb_iterator

flags = tf.flags

flags.DEFINE_string("data_path", "./",
                    "Directory containing PTB raw data (e.g., ptb.train.txt). "
                    "E.g., ./simple-examples/data. If not exists, "
                    "the directory will be created and PTB raw data will "
                    "be downloaded.")
flags.DEFINE_string("config", "config", "The config to use.")

FLAGS = flags.FLAGS

config = importlib.import_module(FLAGS.config)


def _main(_):
    # Data
    batch_size = config.batch_size
    memory_size = config.memory_size
    terminating_learning_rate = config.terminating_learning_rate
    data = prepare_data(FLAGS.data_path)
    vocab_size = data["vocab_size"]
    print('vocab_size = {}'.format(vocab_size))

    inputs = tf.placeholder(tf.int32, [None, memory_size], name="inputs")
    targets = tf.placeholder(tf.int32, [None], name="targets")

    # Model architecture
    initializer = tf.random_normal_initializer(
        stddev=config.initialize_stddev)
    with tf.variable_scope("model", initializer=initializer):
        memnet = tx.modules.MemNetRNNLike(raw_memory_dim=vocab_size,
                                          hparams=config.memnet)
        queries = tf.fill([tf.shape(inputs)[0], config.dim],
                          config.query_constant)
        logits = memnet(inputs, queries)

    # Losses & train ops
    mle_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=targets, logits=logits)
    mle_loss = tf.reduce_sum(mle_loss)

    # Use global_step to pass epoch, for lr decay
    lr = config.opt["optimizer"]["kwargs"]["learning_rate"]
    learning_rate = tf.placeholder(tf.float32, [], name="learning_rate")
    global_step = tf.Variable(0, dtype=tf.int32, name="global_step")
    increment_global_step = tf.assign_add(global_step, 1)
    train_op = tx.core.get_train_op(
        mle_loss,
        learning_rate=learning_rate,
        global_step=global_step,
        increment_global_step=False,
        hparams=config.opt)

    def _run_epoch(sess, data_iter, epoch, is_train=False):
        loss = 0.
        iters = 0

        fetches = {
            "mle_loss": mle_loss
        }
        if is_train:
            fetches["train_op"] = train_op

        mode = (tf.estimator.ModeKeys.TRAIN
                if is_train
                else tf.estimator.ModeKeys.EVAL)

        for _, (x, y) in enumerate(data_iter):
            batch_size = x.shape[0]
            feed_dict = {
                inputs: x, targets: y, learning_rate: lr,
                tx.global_mode(): mode,
            }

            rets = sess.run(fetches, feed_dict)
            loss += rets["mle_loss"]
            iters += batch_size

        ppl = np.exp(loss / iters)
        return ppl

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        try:
            saver.restore(sess, "ckpt/model.ckpt")
            print('restored checkpoint.')
        except BaseException:
            print('restore checkpoint failed.')

        last_valid_ppl = None
        heuristic_lr_decay = (hasattr(config, 'heuristic_lr_decay')
                              and config.heuristic_lr_decay)
        while True:
            if lr < terminating_learning_rate:
                break

            epoch = sess.run(global_step)
            if epoch >= config.num_epochs:
                print('Too many epochs!')
                break

            print('epoch: {} learning_rate: {:.6f}'.format(epoch, lr))

            # Train
            train_data_iter = ptb_iterator(
                data["train_text_id"], batch_size, memory_size)
            train_ppl = _run_epoch(
                sess, train_data_iter, epoch, is_train=True)
            print("Train Perplexity: {:.3f}".format(train_ppl))
            sess.run(increment_global_step)

            # checkpoint
            if epoch % 5 == 0:
                try:
                    saver.save(sess, "ckpt/model.ckpt")
                    print("saved checkpoint.")
                except BaseException:
                    print("save checkpoint failed.")

            # Valid
            valid_data_iter = ptb_iterator(
                data["valid_text_id"], batch_size, memory_size)
            valid_ppl = _run_epoch(sess, valid_data_iter, epoch)
            print("Valid Perplexity: {:.3f}".format(valid_ppl))

            # Learning rate decay
            if last_valid_ppl:
                if heuristic_lr_decay:
                    if valid_ppl > last_valid_ppl * config.heuristic_threshold:
                        lr /= 1. + (valid_ppl / last_valid_ppl
                                    - config.heuristic_threshold) \
                              * config.heuristic_rate
                    last_valid_ppl = last_valid_ppl \
                                     * (1 - config.heuristic_smooth_rate) \
                                     + valid_ppl * config.heuristic_smooth_rate
                else:
                    if valid_ppl > last_valid_ppl:
                        lr /= config.learning_rate_anneal_factor
                    last_valid_ppl = valid_ppl
            else:
                last_valid_ppl = valid_ppl
            print("last_valid_ppl: {:.6f}".format(last_valid_ppl))

        epoch = sess.run(global_step)
        print('Terminate after epoch ', epoch)

        # Test
        test_data_iter = ptb_iterator(data["test_text_id"], 1, memory_size)
        test_ppl = _run_epoch(sess, test_data_iter, 0)
        print("Test Perplexity: {:.3f}".format(test_ppl))


if __name__ == '__main__':
    tf.app.run(main=_main)
