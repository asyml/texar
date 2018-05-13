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
$ python lm_ptb.py --data_path=simple-examples/data --config=config_large
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name, no-member, too-many-locals

import os
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


def _main(_):
    # Data
    batch_size = tf.placeholder(dtype=tf.int32, shape=())
    num_steps = config.num_steps
    data = prepare_data(FLAGS.data_path)
    vocab_size = data["vocab_size"]

    opt_vars = {
        'learning_rate': 0.003,
        'best_valid_ppl': 1e100,
        'steps_not_improved': 0
    }

    inputs = tf.placeholder(tf.int32, [None, num_steps])
    targets = tf.placeholder(tf.int32, [None, num_steps])

    model = EmbeddingTiedLanguageModel(vocab_size=vocab_size)
    initial_state, logits, final_state = \
        model(text_ids=inputs, num_steps=config.num_steps * tf.ones((batch_size, ), dtype=tf.int32))

    # Losses & train ops
    mle_loss = tx.losses.sequence_sparse_softmax_cross_entropy(
        labels=targets,
        logits=logits,
        sequence_length=num_steps * tf.ones((batch_size, )))

    l2_loss = sum([tf.nn.l2_loss(t) for t in tf.trainable_variables()])

    # Use global_step to pass epoch, for lr decay
    global_step = tf.Variable(0, dtype=tf.int32)
    learning_rate = \
        tf.placeholder(dtype=tf.float32, shape=(), name='learning_rate')
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        beta1=0.,
        beta2=0.999,
        epsilon=1e-9)
    train_op = optimizer.minimize(
        mle_loss + config.l2_decay * l2_loss, global_step=global_step)

    def _run_epoch(sess, data_iter, mode_string):
        if mode_string == 'train':
            cur_batch_size = config.training_batch_size
        elif mode_string == 'valid':
            cur_batch_size = config.valid_batch_size
        elif mode_string == 'test':
            cur_batch_size = config.test_batch_size

        start_time = time.time()
        loss = 0.
        iters = 0
        state = sess.run(initial_state, feed_dict={
            inputs: np.ones((cur_batch_size, config.num_steps))})

        fetches = {
            "mle_loss": mle_loss,
            "final_state": final_state,
            'global_step': global_step
        }
        if mode_string == 'train':
            fetches["train_op"] = train_op

        mode = (tf.estimator.ModeKeys.TRAIN if mode_string=='train'
                else tf.estimator.ModeKeys.EVAL)
        epoch_size = (len(data) // batch_size - 1) // num_steps
        for step, (x, y) in enumerate(data_iter):
            feed_dict = {
                batch_size: cur_batch_size,
                inputs: x, targets: y,
                learning_rate: opt_vars['learning_rate'],
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

            if mode_string == 'train':
                print('global step:', rets['global_step'], ' ' * 4,
                      'training ppl:', ppl,
                      file=training_log)
                training_log.flush()

            if mode_string == 'train' and rets['global_step'] % 100 == 0:
                valid_data_iter = ptb_iterator(
                    data["valid_text_id"], config.valid_batch_size, num_steps)
                valid_ppl = \
                    _run_epoch(sess, valid_data_iter, mode_string='valid')
                test_data_iter = ptb_iterator(
                    data["test_text_id"], config.test_batch_size, num_steps)
                test_ppl = _run_epoch(sess, test_data_iter, mode_string='test')
                print('global step:', rets['global_step'], ' ' * 4,
                      'learning rate:', opt_vars['learning_rate'], ' ' * 4,
                      'valid ppl:', valid_ppl, ' ' * 4,
                      'test ppl:', test_ppl,
                      file=eval_log)
                eval_log.flush()

                if valid_ppl < opt_vars['best_valid_ppl']:
                    opt_vars['best_valid_ppl'] = valid_ppl
                    opt_vars['steps_not_improved'] = 0
                else:
                    opt_vars['steps_not_improved'] += 1

                if opt_vars['steps_not_improved'] >= 30:
                    opt_vars['steps_not_improved'] = 0
                    opt_vars['learning_rate'] *= config.lr_decay

        ppl = np.exp(loss / iters)
        return ppl

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        for epoch in range(config.num_epochs):
            # Train
            train_data_iter = ptb_iterator(
                data["train_text_id"], config.training_batch_size, num_steps)
            train_ppl = _run_epoch(sess, train_data_iter, mode_string='train')
            print("Epoch: %d Train Perplexity: %.3f" % (epoch, train_ppl))
            # Valid
            valid_data_iter = ptb_iterator(
                data["valid_text_id"], config.valid_batch_size, num_steps)
            valid_ppl = _run_epoch(sess, valid_data_iter, mode_string='valid')
            print("Epoch: %d Valid Perplexity: %.3f" % (epoch, valid_ppl))
            # Test
            test_data_iter = ptb_iterator(
                data["test_text_id"], config.test_batch_size, num_steps)
            test_ppl = _run_epoch(sess, test_data_iter, mode_string='test')
            print("Test Perplexity: %.3f" % (test_ppl))


if __name__ == '__main__':
    LOG_DIR = 'language_model_training_log/'
    os.system('mkdir ' + LOG_DIR)

    training_log = open(LOG_DIR + 'training_log.txt', 'w')
    eval_log = open(LOG_DIR + 'eval_log.txt', 'w')
    tf.app.run(main=_main)
