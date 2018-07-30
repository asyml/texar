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
"""Example for building a sentence convolutional classifier.

Use `./sst_data_preprocessor.py` to download and clean the SST binary data.

To run:

$ python clas_main.py --config=config_kim
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import tensorflow as tf
import texar as tx

# pylint: disable=invalid-name, too-many-locals

flags = tf.flags

flags.DEFINE_string("config", "config_kim", "The config to use.")

FLAGS = flags.FLAGS

config = importlib.import_module(FLAGS.config)

def _main(_):
    # Data
    train_data = tx.data.MultiAlignedData(config.train_data)
    val_data = tx.data.MultiAlignedData(config.val_data)
    test_data = tx.data.MultiAlignedData(config.test_data)
    iterator = tx.data.TrainTestDataIterator(train_data, val_data, test_data)
    batch = iterator.get_next()

    # Model architecture
    embedder = tx.modules.WordEmbedder(
        vocab_size=train_data.vocab('x').size, hparams=config.emb)
    classifier = tx.modules.Conv1DClassifier(config.clas)
    logits, pred = classifier(embedder(batch['x_text_ids']))

    # Losses & train ops
    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=batch['y'], logits=logits)
    accu = tx.evals.accuracy(batch['y'], pred)

    train_op = tx.core.get_train_op(loss, hparams=config.opt)

    def _run_epoch(sess, mode, epoch=0, verbose=False):
        is_train = tx.utils.is_train_mode_py(mode)

        fetches = {
            "accu": accu,
            "batch_size": tx.utils.get_batch_size(batch['y'])
        }
        if is_train:
            fetches["train_op"] = train_op
        feed_dict = {tx.context.global_mode(): mode}

        cum_accu = 0.
        nsamples = 0
        step = 0
        while True:
            try:
                rets = sess.run(fetches, feed_dict)
                step += 1

                accu_ = rets['accu']
                cum_accu += accu_ * rets['batch_size']
                nsamples += rets['batch_size']

                if verbose and (step == 1 or step % 100 == 0):
                    tf.logging.info(
                        "epoch: {0:2} step: {1:4} accu: {2:.4f}"
                        .format(epoch, step, accu_))
            except tf.errors.OutOfRangeError:
                break
        return cum_accu / nsamples

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        best_val_accu = -1.
        for epoch in range(config.num_epochs):
            # Train
            iterator.switch_to_train_data(sess)
            train_accu = _run_epoch(sess, tf.estimator.ModeKeys.TRAIN, epoch)
            # Val
            iterator.switch_to_val_data(sess)
            val_accu = _run_epoch(sess, tf.estimator.ModeKeys.EVAL, epoch)
            tf.logging.info('epoch: {0:2} train accu: {1:.4f} val accu: {2:.4f}'
                            .format(epoch+1, train_accu, val_accu))
            # Test
            if val_accu > best_val_accu:
                best_val_accu = val_accu

                iterator.switch_to_test_data(sess)
                test_accu = _run_epoch(sess, tf.estimator.ModeKeys.EVAL)
                tf.logging.info('test accu: {0:.4f}'.format(test_accu))

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=_main)
