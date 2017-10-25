#
"""
Example pipeline. This is a minimal example of basic RNN language model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name, no-name-in-module

import tensorflow as tf
from tensorflow.python.framework import ops

# We shall wrap all these modules
from txtgen.data import database
from txtgen.modules import ConstantConnector
from txtgen.modules import BasicRNNDecoder, get_helper
from txtgen.losses import mle_losses
from txtgen.core import optimization as opt
from txtgen import context

if __name__ == "__main__":
    ### Build data pipeline

    # Config data hyperparams. Hyperparams not configured will be automatically
    # filled with default values. For text database, default values are defined
    # in `txtgen.data.database.default_text_dataset_hparams()`.
    data_hparams = {
        "num_epochs": 10,
        "seed": 123,
        "dataset": {
            "files": ['data/sent.txt'],
            "vocab_file": 'data/vocab.txt'
        }
    }
    # Construct the database
    text_db = database.MonoTextDataBase(data_hparams)
    # Get data minibatch, which is a dictionary:
    # {
    #   "text": text_tensor,     # text string minibatch,
    #   "length": length_tensor, # a 1D tensor of sequence lengths with
    #                            # shape `[batch_size]`,
    #   "text_ids": text_id_tensor, # a 2D int tensor of token ids with shape
    #                               # `[batch_size, max_seq_length]`
    # }
    data_batch = text_db()

    ### Build model

    # Build decoder. Simply use the default hyperparameters.
    #decoder = rnn_decoders.BasicRNNDecoder(vocab_size=text_db.vocab.vocab_size)
    decoder = BasicRNNDecoder(vocab_size=text_db.vocab.vocab_size)

    # Build connector, which simply feeds zero state to decoder as initial state
    connector = ConstantConnector(decoder.state_size)

    # Build helper used in training.
    # We shall probably improve the interface here.
    helper_train = get_helper(
        decoder.hparams.helper_train.type,
        inputs=data_batch['text_ids'][:, :-1],
        sequence_length=data_batch['length'] - 1,
        embedding=decoder.embedding)

    # Decode
    outputs, final_state, sequence_lengths = decoder(
        helper=helper_train, initial_state=connector(text_db.batch_size))

    # Build loss
    mle_loss = mle_losses.average_sequence_sparse_softmax_cross_entropy(
        labels=data_batch['text_ids'][:, 1:],
        logits=outputs.rnn_output,
        sequence_length=sequence_lengths - 1)

    # Build train op. Only config the optimizer while using default settings
    # for other hyperparameters.
    opt_hparams = {
        "optimizer": {
            "type": "MomentumOptimizer",
            "kwargs": {
                "learning_rate": 0.01,
                "momentum": 0.9
            }
        }
    }
    train_op, global_step = opt.get_train_op(mle_loss, hparams=opt_hparams)

    ### Graph is done. Now start running

    # We shall wrap these environment setup codes
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        #with tf.contrib.slim.queues.QueueRunners(sess):
        #    _, step, loss = sess.run(
        #        [train_op, global_step, mle_loss],
        #        feed_dict={context.is_train(): True})

        #    if step % 10 == 0:
        #        print("%d: %.6f" % (step, loss))


        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        print(ops.get_collection(ops.GraphKeys.QUEUE_RUNNERS))

        try:
            while not coord.should_stop():
                # Run the logics

                _, step, loss = sess.run(
                    [train_op, global_step, mle_loss],
                    feed_dict={context.is_train(): True})

                if step % 10 == 0:
                    print("%d: %.6f" % (step, loss))

        except tf.errors.OutOfRangeError:
            print('Done -- epoch limit reached')
        finally:
            coord.request_stop()
        coord.join(threads)

