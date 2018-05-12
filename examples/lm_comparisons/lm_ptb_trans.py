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

    embedder = tx.modules.WordEmbedder(
        vocab_size=vocab_size, hparams=config.emb)

    decoder = tx.modules.TransformerDecoder(
        embedding=embedder._embedding,
        hparams=config.decoder_hparams)

    logits, preds = decoder(
        decoder_input=inputs,
        encoder_output=None, #there should be 1-pos bias
        encoder_decoder_attention_bias=None,
    )
    predictions = decoder.dynamic_decode(
        emb_inputs=inputs,
        encoder_decoder_attention_bias=None,
    )

    # Losses & train ops
    mle_loss = tx.losses.sequence_sparse_softmax_cross_entropy(
        labels=targets,
        logits=logits,
        sequence_length=[num_steps] * batch_size)

    #l2_loss = sum([tf.nn.l2_loss(t) for t in tf.trainable_variables])
    # should we add the l2_loss on language model?

    global_step = tf.Variable(0, dtype=tf.int32)
    learning_rate = \
        tf.placeholder(dtype=tf.float32, shape=(), name='learning_rate')
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        beta1=config.opt_hparams['Adam_beta1'],
        beta2=config.opt_hparams['Adam_beta2'],
        epsilon=1e-9)
    train_op = optimizer.minimize(mle_loss, global_step=global_step)

    def _run_epoch(sess, data_iter, is_train=False, verbose=True):
        start_time = time.time()
        loss = 0.
        iters = 0
        fetches = {
            "mle_loss": mle_loss,
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
            #for i, (c, h) in enumerate(initial_state):
            #    feed_dict[c] = state[i].c
            #    feed_dict[h] = state[i].h

            rets = sess.run(fetches, feed_dict)
            loss += rets["mle_loss"]
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
                sess, train_data_iter, is_train=True, verbose=True)
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
