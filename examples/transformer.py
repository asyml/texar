"""
Example pipeline. This is a minimal example of basic RNN language model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name, no-name-in-module

import tensorflow as tf
tf.set_random_seed(123)
import random
random.seed(123)
import numpy as np
np.random.seed(123)

# We shall wrap all these modules
from txtgen.data import PairedTextDataBase
# from txtgen.modules import ConstantConnector
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
        # "shuffle":False,
        "source_dataset": {
            "files": ['data/translation/de-en/train_de_sentences.txt'],
            "vocab_file": 'data/translation/de-en/filter_de.vocab.txt',
            "processing":{
                "bos_token": "<S>",
                "eos_token": "</S>",
                # max_seq_length
                }
        },
        "target_dataset": {
            "files": ['data/translation/de-en/train_en_sentences.txt'],
            "vocab_file": 'data/translation/de-en/filter_en.vocab.txt',
            "processing":{
                "bos_token": "<S>",
                "eos_token": "</S>",
            },
        }
    }
    extra_hparams = {
        'max_seq_length':10,
        'scale':True,
        'sinusoid':False,
        'embedding': {
            'initializer': {
                'type':'xavier_initializer',
                },
            'dim': 512,
        },
        'num_blocks': 6,
        'num_heads': 8,
    }
    # Construct the database
    text_database = PairedTextDataBase(data_hparams)
    print('database finished')

    text_data_batch = text_database()
    encoder = TransformerEncoder(vocab_size=text_database.source_vocab.vocab_size,
            hparams=extra_hparams)
    print('encoder scope:{}'.format(encoder.variable_scope.name))
    decoder = TransformerDecoder(vocab_size=text_database.target_vocab.vocab_size,
            hparams=extra_hparams)

    ori_src_text = text_data_batch['source_text_ids']

    ori_tgt_text = text_data_batch['target_text_ids']

    padded_src_text = tf.concat(
            [ori_src_text, tf.zeros([tf.shape(ori_src_text)[0],
            extra_hparams['max_seq_length']+1-tf.shape(ori_src_text)[1]], dtype=tf.int64)], axis=1)

    padded_tgt_text = tf.concat(
            [ori_tgt_text, tf.zeros([tf.shape(ori_tgt_text)[0],
                extra_hparams['max_seq_length']+1-tf.shape(ori_tgt_text)[1]], dtype=tf.int64)], axis=1)

    encoder_input = padded_src_text[:, 1:]
    decoder_input = padded_tgt_text[:, :-1]

    encoder_output = encoder(encoder_input)
    logits, preds = decoder(decoder_input, encoder_output)

    loss_params = {
            'label_smoothing':0.1,
    }

    labels = padded_tgt_text[:, 1:]
    is_target = tf.to_float(tf.not_equal(labels, 0))
    acc = tf.reduce_sum(tf.to_float(tf.equal(preds, labels))*is_target)/ (tf.reduce_sum(is_target))

    labels = mle_losses.label_smoothing(labels,
            text_database.target_vocab.vocab_size,\
            loss_params['label_smoothing'])

    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    mean_loss = tf.reduce_sum(loss*is_target) / (tf.reduce_sum(is_target))

    #mle_loss = mle_losses.average_sequence_softmax_cross_entropy(
    #       labels=labels,
    #       logits=logits,
    #       sequence_length=text_data_batch['target_length']-1)
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
    train_op, global_step = opt.get_train_op(mean_loss, hparams=opt_hparams)
    saver = tf.train.Saver()
    # We shall wrap these environment setup codes

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # print(ops.get_collection(ops.GraphKeys.QUEUE_RUNNERS))

        try:
            while not coord.should_stop():
                #pred loss, not include BOS
                #source, target, _, step, loss, accy= sess.run(
                #    [padded_src_text, padded_tgt_text, train_op, global_step, mean_loss, acc],
                #    feed_dict={context.is_train(): True})
                source, target = sess.run(
                        [padded_src_text, padded_tgt_text], feed_dict={context.is_train():True})

                print('source:{}\n target:{}\n'.format(source, target))
                # for var in tf.trainable_variables():
                #     print('name:{}\tshape:{}\ttype:{}\n'.format(var.name, var.shape, var.dtype))
                if step % 100 == 0:
                    print("time{} step{} loss{} acc{}".format(time.asctime(time.localtime(time.time())), step, loss, accy))
                if step % 1000 == 0:
                    saver.save(sess, './logdir/my-model', global_step = step)
                # if step % 2000 ==0:
                #    coord.request_stop()
        except tf.errors.OutOfRangeError:
            print('Done -- epoch limit reached')
        finally:
            coord.request_stop()
        coord.join(threads)

