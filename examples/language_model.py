#
"""
Example pipeline. This is a minimal example of basic RNN language model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# We shall wrap all these modules
from txtgen.data import database
from txtgen.modules.connectors import connectors
from txtgen.modules.decoders import rnn_decoders, rnn_decoder_helpers
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
            "vocab.file": 'data/vocab.txt'
        }
    }
    # Construct the database
    text_db = database.TextDataBase(data_hparams)
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
    decoder = rnn_decoders.BasicRNNDecoder(vocab_size=text_db.vocab.vocab_size)

    # Build connector, which simply feeds zero state to decoder as initial state
    connector = connectors.ConstantConnector(decoder.state_size)

    # Build helper used in training.
    # We shall probably improve the interface here.
    helper_train = rnn_decoder_helpers.get_helper(
        decoder.hparams.helper_train.type,
        inputs=data_batch['text_ids'],
        sequence_length=data_batch['length'],
        embedding=decoder.embedding)

    # Decode
    outputs, final_state, sequence_lengths = decoder(
        helper=helper_train, initial_state=connector(text_db.batch_size))

    # Build loss
    mle_loss = mle_losses.average_sequence_sparse_softmax_cross_entropy(
        labels=data_batch['text_ids'],
        logits=outputs.rnn_output,
        sequence_length=sequence_lengths)

    # Build train op. Simply use default hyperparameter setting.
    train_op, global_step = opt.get_train_op(mle_loss)


    ### Graph is done. Now start running

    # We shall wrap these environment setup codes
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                # Run the logics

                _, step, loss = sess.run(
                    [train_op, global_step, mle_loss],
                    feed_dict={context.is_train(): True})

                if step % 100 == 0:
                    print("%d: %.6f" % (step, loss))

        except tf.errors.OutOfRangeError:
            print('Done -- epoch limit reached')
        finally:
            coord.request_stop()
        coord.join(threads)

