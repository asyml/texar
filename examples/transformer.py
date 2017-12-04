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
from txtgen.data import PairedTextDataBase
from txtgen.modules import ConstantConnector
from txtgen.modules import TransformerEncoder, TransformerDecoder
from txtgen.losses import mle_losses
from txtgen.core import optimization as opt
from txtgen import context
import time

if __name__ == "__main__":
    ### Build data pipeline

    # Config data hyperparams. Hyperparams not configured will be automatically
    # filled with default values. For text database, default values are defined
    # in `txtgen.data.database.default_text_dataset_hparams()`.
    data_hparams = {
        "num_epochs": 20,
        "seed": 123,
        "batch_size":32,
        "source_dataset": {
            "files": ['data/translation/de-en/train_de_sentences.txt'],
            "vocab_file": 'data/translation/de-en/filter_de.vocab.txt',
            "processing":{
                "bos_token": "<SOURCE_BOS>",
                "eos_token": "<SOURCE_EOS>",
                }
        },
        "target_dataset": {
            "files": ['data/translation/de-en/train_en_sentences.txt'],
            "vocab_file": 'data/translation/de-en/filter_en.vocab.txt',
            # "reader_share": True,
            "processing":{
                "bos_token": "<TARGET_BOS>",
                "eos_token": "<TARGET_EOS>",
            },
        }
    }
    extra_hparams = {
        'max_seq_length':10,
        'scale':True,
        'sinusoid':False,
        'embedding': {
            'dim': 512,
        },
        'num_blocks': 6,
        'num_heads': 8,
    }
    # Construct the database
    text_database = PairedTextDataBase(data_hparams)
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
    ori_src_text = text_data_batch['source_text_ids']
    ori_src_words = text_database.source_vocab.id_to_token_map.lookup(tf.to_int64(ori_src_text))
    src_text = ori_src_text[:, :extra_hparams['max_seq_length']+1]
    # src_Text : no need to add <BOS>

    ori_tgt_text = text_data_batch['target_text_ids']
    ori_tgt_words = text_database.target_vocab.id_to_token_map.lookup(tf.to_int64(ori_tgt_text))
    tgt_text = ori_tgt_text[:, :extra_hparams['max_seq_length']+1]
    # decoder_inputs = tgt_text[:, :-1]
    # tgt_text : need to add <BOS>, no need to add <EOS>
    # but [:-1] cannot delete the EOS completely

    padded_src_text = tf.concat(
            [src_text[:, 1:], tf.zeros([tf.shape(src_text)[0],
            extra_hparams['max_seq_length']+1-tf.shape(src_text)[1]], dtype=tf.int64)], axis=1)

    padded_tgt_text = tf.concat(
            [tgt_text[:, :-1], tf.zeros([tf.shape(tgt_text)[0],
                extra_hparams['max_seq_length']+1-tf.shape(tgt_text)[1]], dtype=tf.int64)], axis=1)
    encoder_output = encoder(padded_src_text)
    logits, preds = decoder(padded_tgt_text, encoder_output)

    loss_params = {
            'label_smoothing':0.1,
    }

    labels = tf.concat(
            [tgt_text[:, 1:], tf.zeros([tf.shape(tgt_text)[0],
                extra_hparams['max_seq_length']+1-tf.shape(tgt_text)[1]], dtype=tf.int64)], axis=1)

    smooth_labels = mle_losses.label_smoothing(labels, text_database.target_vocab.vocab_size, loss_params['label_smoothing'])
    print('smooth_labels:{}'.format(smooth_labels.shape))
    print('logits:{}'.format(logits.shape))
    mle_loss = mle_losses.average_sequence_softmax_cross_entropy(
            labels=smooth_labels,
            logits=logits,
            sequence_length=text_data_batch['target_length']-1)
    # target_length, should not include BOS

    tf.summary.scalar('mean_loss', mle_loss)
    merged = tf.summary.merge_all()

    # Build train op. Only config the optimizer while using default settings
    # for other hyperparameters.
    opt_hparams={
        "optimizer": {
            "type": "AdamOptimizer",
            "kwargs": {
                "learning_rate": 0.0001,
                "beta1": 0.9,
                "beta2": 0.98,
                "epsilon": 1e-8,
            }
        }
    }
    train_op, global_step = opt.get_train_op(mle_loss, hparams=opt_hparams)

    ### Graph is done. Now start running
    saver = tf.train.Saver()
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

                #pred loss, not include BOS
                source, target, _, step, loss= sess.run(
                    [ori_src_words, ori_tgt_words, train_op, global_step, mle_loss],
                    feed_dict={context.is_train(): True})
                if source.shape[1]>11 or target.shape[1]>11:
                    print('source:{}\n target:{}\n'.format(source, target))

                if step % 100 == 0:
                    print("time{} step{} loss{}".format(time.asctime(time.localtime(time.time())), step, loss))
                if step % 1000 == 0:
                    saver.save(sess, './logdir/my-model', global_step = step)
                # if step % 2000 ==0:
                #    coord.request_stop()
        except tf.errors.OutOfRangeError:
            print('Done -- epoch limit reached')
        finally:
            coord.request_stop()
        coord.join(threads)

