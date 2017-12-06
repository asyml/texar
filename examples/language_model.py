#
"""
Example pipeline. This is a minimal example of basic RNN language model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name, no-name-in-module

# question: the loaded data is different each time I run it.
# even if I have set random seeds for tensorflow, numpy and random,
# and I have manually set the seed for database instance.

rseed=123
import sys

import tensorflow as tf
tf.set_random_seed(rseed)
import random
random.seed(rseed)
import numpy as np
np.random.seed(rseed)

# We shall wrap all these modules
from txtgen.data import MonoTextDataBase
from txtgen.modules import ConstantConnector
from txtgen.modules import BasicRNNDecoder, get_helper
from txtgen.losses import mle_losses
from txtgen.core import optimization as opt
from txtgen import context



def load_data():
    # Build data pipeline

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
    text_db = MonoTextDataBase(data_hparams)
    # Get data minibatch, which is a dictionary:
    # {
    #   "text": text_tensor,     # text string minibatch,
    #   "length": length_tensor, # a 1D tensor of sequence lengths with
    #                            # shape `[batch_size]`,
    #   "text_ids": text_id_tensor, # a 2D int tensor of token ids with shape
    #                               # `[batch_size, max_seq_length]`
    # }
    data_batch = text_db()
    return data_batch, text_db.vocab, text_db.batch_size


def create_model(vocab_size):
    # Ensure model structure for train and inference.
    # Build decoder. Simply use the default hyperparameters.
    return BasicRNNDecoder(vocab_size=vocab_size)


def train():
    data_batch, vocab, batch_size = load_data()

    decoder = create_model(vocab.vocab_size)

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
        helper=helper_train, initial_state=connector(batch_size))

    print('decoder done')
    # Build loss
    mle_loss = mle_losses.average_sequence_sparse_softmax_cross_entropy(
        labels=data_batch['text_ids'][:, 1:],
        logits=outputs.rnn_output,
        sequence_length=sequence_lengths)

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

        # with tf.contrib.slim.queues.QueueRunners(sess):
        #    _, step, loss = sess.run(
        #        [train_op, global_step, mle_loss],
        #        feed_dict={context.is_train(): True})

        #    if step % 10 == 0:
        #        print("%d: %.6f" % (step, loss))

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                # Run the logics

                _, step, loss = sess.run(
                    [train_op, global_step, mle_loss],
                    feed_dict={context.is_train(): True})

                sequence_length_, seq_len = sess.run(
                    [sequence_lengths, data_batch['length']],
                    feed_dict={context.is_train(): True})

                # question: can we run the sess.run twice?
                # I'm wondering whether the loaded data will be different in these two times

                print('sep len:{}'.format(seq_len))
                if step % 10 == 0:
                    print("%d: %.6f" % (step, loss))

        except tf.errors.OutOfRangeError:
            print('Done -- epoch limit reached')
        finally:
            coord.request_stop()
        coord.join(threads)

        saver = tf.train.Saver(tf.global_variables())
        saver.save(sess, './checkpoints/lm.ckpt', global_step=global_step)


def sample(ckpt, n_samples=5):
    _, vocab, _ = load_data()

    decoder = create_model(vocab.vocab_size)

    # Build connector, which simply feeds zero state to decoder as initial state
    connector = ConstantConnector(decoder.state_size)

    # Build the inference helper.
    helper_infer = get_helper(
        decoder.hparams.helper_infer.type,
        embedding=decoder.embedding,
        start_tokens=[vocab._token_to_id_map_py[vocab.bos_token]] * n_samples,
        end_token=vocab._token_to_id_map_py[vocab.eos_token],
        softmax_temperature=None,
        seed=None
    )

    # Decode
    outputs, final_state, sequence_lengths = decoder(
        helper=helper_infer, initial_state=connector(n_samples))

    with tf.Session() as sess:
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(sess, ckpt)

        sess_outputs = sess.run(outputs, feed_dict={context.is_train(): False})
        for i in range(n_samples):
            words = [vocab._id_to_token_map_py[id] for id in sess_outputs.sample_id[i]]

            if vocab.eos_token in words:
                words = words[:words.index(vocab.eos_token)]

            print('> ' + ' '.join(words))


if __name__ == "__main__":

    ckpt = sys.argv[1] if len(sys.argv) > 1 else None  # model checkpoint path

    if ckpt:
        sample(ckpt)
    else:
        train()
