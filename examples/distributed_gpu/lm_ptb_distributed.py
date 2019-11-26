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

# pylint: disable=invalid-name, no-member, too-many-locals

import time
import importlib
import numpy as np
import tensorflow as tf
import texar.tf as tx
import horovod.tensorflow as hvd

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
    tf.logging.set_verbosity(tf.logging.INFO)

    # 1. initialize the horovod
    hvd.init()

    batch_size = config.batch_size
    num_steps = config.num_steps
    data = prepare_data(FLAGS.data_path)
    vocab_size = data["vocab_size"]

    inputs = tf.placeholder(tf.int32, [None, num_steps],
                            name='inputs')
    targets = tf.placeholder(tf.int32, [None, num_steps],
                             name='targets')

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

        # This _batch_size equals to batch_size // hvd.size() in
        # distributed training.
        # because the mini-batch is distributed to multiple GPUs

        _batch_size = tf.shape(inputs)[0]
        initial_state = decoder.zero_state(_batch_size,
                                           tf.float32)
        seq_length = tf.broadcast_to([num_steps], (_batch_size, ))
        outputs, final_state, seq_lengths = decoder(
            decoding_strategy="train_greedy",
            impute_finished=True,
            inputs=emb_inputs,
            sequence_length=seq_length,
            initial_state=initial_state)
    # Losses & train ops
    mle_loss = tx.losses.sequence_sparse_softmax_cross_entropy(
        labels=targets,
        logits=outputs.logits,
        sequence_length=seq_lengths)

    # Use global_step to pass epoch, for lr decay
    global_step = tf.placeholder(tf.int32)

    opt = tx.core.get_optimizer(
        global_step=global_step,
        hparams=config.opt
    )

    # 2. wrap the optimizer
    opt = hvd.DistributedOptimizer(opt)

    train_op = tx.core.get_train_op(
        loss=mle_loss,
        optimizer=opt,
        global_step=global_step,
        learning_rate=None,
        increment_global_step=False,
        hparams=config.opt
    )

    def _run_epoch(sess, data_iter, epoch, is_train=False, verbose=False):
        start_time = time.time()
        loss = 0.
        iters = 0

        fetches = {
            "mle_loss": mle_loss,
            "final_state": final_state,
        }
        if is_train:
            fetches["train_op"] = train_op
            epoch_size = (len(data["train_text_id"]) // batch_size - 1)\
                // num_steps

        mode = (tf.estimator.ModeKeys.TRAIN
                if is_train
                else tf.estimator.ModeKeys.EVAL)

        for step, (x, y) in enumerate(data_iter):
            if step == 0:
                state = sess.run(initial_state,
                                 feed_dict={inputs: x})

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
            if verbose and is_train and hvd.rank() == 0 \
                and (step + 1) % (epoch_size // 10) == 0:
                tf.logging.info("%.3f perplexity: %.3f speed: %.0f wps" %
                                ((step + 1) * 1.0 / epoch_size, ppl,
                                 iters * batch_size / (
                                         time.time() - start_time)))
        _elapsed_time = time.time() - start_time
        tf.logging.info("epoch time elapsed: %f" % (_elapsed_time))
        ppl = np.exp(loss / iters)
        return ppl, _elapsed_time

    # 3. set broadcase global variables from rank-0 process
    bcast = hvd.broadcast_global_variables(0)

    # 4. set visible GPU
    session_config = tf.ConfigProto()
    session_config.gpu_options.visible_device_list = str(hvd.local_rank())

    with tf.Session(config=session_config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        # 5. run the broadcast_global_variables node before training
        bcast.run()

        _times = []
        for epoch in range(config.num_epochs):
            # Train
            train_data_iter = ptb_iterator(
                data["train_text_id"], config.batch_size, num_steps,
                is_train=True)
            train_ppl, train_time = _run_epoch(
                sess, train_data_iter, epoch, is_train=True, verbose=True)
            _times.append(train_time)
            tf.logging.info("Epoch: %d Train Perplexity: %.3f" % (epoch, train_ppl))
            # Valid in the main process
            if hvd.rank() == 0:
                valid_data_iter = ptb_iterator(
                    data["valid_text_id"], config.batch_size, num_steps)
                valid_ppl, _ = _run_epoch(sess, valid_data_iter, epoch)
                tf.logging.info("Epoch: %d Valid Perplexity: %.3f"
                                % (epoch, valid_ppl))

        tf.logging.info('train times: %s' % (_times))
        tf.logging.info('average train time/epoch %f'
                        % np.mean(np.array(_times)))
        # Test in the main process
        if hvd.rank() == 0:
            test_data_iter = ptb_iterator(
                data["test_text_id"], batch_size, num_steps)
            test_ppl, _ = _run_epoch(sess, test_data_iter, 0)
            tf.logging.info("Test Perplexity: %.3f" % (test_ppl))


if __name__ == '__main__':
    tf.app.run(main=_main)
