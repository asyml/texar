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
from txtgen.modules import TransformerEncoder, TransformerDecoder
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
        "batch_size":32,
        "source_dataset": {
            "files": ['data/translation/de-en/de_sentences.txt'],
            "vocab_file": 'data/translation/de-en/de.vocab.txt',
            "processing":{
                "bos_token": "<SOURCE_BOS>",
                "eos_token": "<SOURCE_EOS>",
                }
        },
        "target_dataset": {
            "files": ['data/translation/de-en/en_sentences.txt'],
            "vocab_file": 'data/translation/de-en/en.vocab.txt',
            # "reader_share": True,
            "processing":{
                "bos_token": "<TARGET_BOS>",
                "eos_token": "<TARGET_EOS>",
            },
        }
    }
    extra_hparams = {
        'max_seq_length':100,
        'scale':100,
        'sinusoid':True,
        'embedding': {
            'dim': 512,
        },
        'num_blocks': 6,
        'num_heads': 8,
    }
    # Construct the database
    text_database = database.PairedTextDataBase(data_hparams)
    print('database finished')

    text_data_batch = text_database()
    # Get data minibatch, which is a dictionary:
    # {
    #   "text": text_tensor,     # text string minibatch,
    #   "length": length_tensor, # a 1D tensor of sequence lengths with
    #                            # shape `[batch_size]`,
    #   "text_ids": text_id_tensor, # a 2D int tensor of token ids with shape
    #                               # `[batch_size, max_seq_length]`
    # }

    # Build decoder. Simply use the default hyperparameters.
    #decoder = rnn_decoders.BasicRNNDecoder(vocab_size=text_db.vocab.vocab_size)
    encoder = TransformerEncoder(vocab_size=text_database.source_vocab.vocab_size,
            hparams=extra_hparams)
    decoder = TransformerDecoder(vocab_size=text_database.target_vocab.vocab_size,
            hparams=extra_hparams)

    # Build connector, which simply feeds zero state to decoder as initial state
    connector = ConstantConnector(output_size=decoder._hparams.embedding.dim)
    print('encoder decoder finished')
    src_text = text_data_batch['source_text_ids']
    tgt_text = text_data_batch['target_text_ids']

    # shifted right
    decoder_inputs = tf.concat((tf.ones_like(tgt_text[:, :1]), tgt_text[:, :-1]), -1)

    # print('src_text:{}'.format(src_text))
    encoder_output = encoder(src_text,
            sequence_length=text_data_batch['source_length'])
    # Decode
    # print('encoder_output:{}'.format(encoder_output.shape))
    logits, preds = decoder(decoder_inputs, encoder_output)
    labels = text_data_batch['target_text_ids']

    # logits_shape = tf.shape(logits)
    # labels_shape = tf.shape(labels)
    lengths = text_data_batch['target_length']-1
    # src_length =tf.shape(text_data_batch['source_length'])
    # target_length = tf.shape(lengths)
    #istarget = tf.to_float(tf.not_equal(y, 0))
    # acc = tf.reduce_sum(tf.to_float(tf.equal(preds,
    # Build loss
    mle_loss = mle_losses.average_sequence_sparse_softmax_cross_entropy(
        labels=labels,
        logits=logits,
        sequence_length=text_data_batch['target_length']-1)

    # Build train op. Only config the optimizer while using default settings
    # for other hyperparameters.
    opt_hparams={
        "optimizer": {
            "type": "MomentumOptimizer",
            "kwargs": {
                "learning_rate": 0.0001,
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
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        print(ops.get_collection(ops.GraphKeys.QUEUE_RUNNERS))

        try:
            while not coord.should_stop():

                # logit_sp, label_sp= sess.run(
                #     [logits_shape, labels_shape],
                #     feed_dict={context.is_train(): True})
                _, step, loss = sess.run(
                        [train_op, global_step, mle_loss],
                        feed_dict={context.is_train():True})
                if step % 10 == 0:
                    print("%d: %.6f" % (step, loss))

        except tf.errors.OutOfRangeError:
            print('Done -- epoch limit reached')
        finally:
            coord.request_stop()
        coord.join(threads)

