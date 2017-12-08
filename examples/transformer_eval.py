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
from texar.data import PairedTextDataBase
# from texar.modules import ConstantConnector
from texar.modules import TransformerEncoder, TransformerDecoder
from texar.losses import mle_losses
from texar.core import optimization as opt
from texar import context

if __name__ == "__main__":
    ### Build data pipeline

    # Config data hyperparams. Hyperparams not configured will be automatically
    # filled with default values. For text database, default values are defined
    # in `texar.data.database.default_text_dataset_hparams()`.
    data_hparams = {
        "num_epochs": 20,
        "seed": 123,
        "batch_size":32,
        "shuffle":False,
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
            'name': 'embedding',
            'dim': 512,
            'initializer': {
                'type': tf.contrib.layers.xavier_initializer(),
                },
            'trainable':True,
            # 'regularizer':

            #    'type':'xavier_initializer',
            #    'kwargs': {
            #        'uniform':True,
            #        'seed':None,
            #        'dtype':tf.float32,
            #        },
            #    },
        },
        'num_blocks': 6,
        'num_heads': 8,
    }
    # Construct the database
    text_database = PairedTextDataBase(data_hparams)
    #dedict = text_database.source_vocab._id_to_token_map_py
    #for idx, word in dedict.items():
    #    print('id:{} word:{}'.format(idx, word))
    # print('database finished')
    text_data_batch = text_database()
    encoder = TransformerEncoder(vocab_size=text_database.source_vocab.vocab_size,
            hparams=extra_hparams)
    print('encoder scope:{}'.format(encoder.variable_scope.name))
    decoder = TransformerDecoder(vocab_size=text_database.target_vocab.vocab_size,
            hparams=extra_hparams)

    target_vocab = text_database.target_vocab
    helper_infer = get_helper(
            decoder.hparams.helper_infer.type,
            embedding = decoder._embedding,
            start_tokens=[target_vocab._token_to_id_map_py[target_vocab.padding_token]*\
                data_hparams['batch_size']])

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

    srcdata = tf.summary.histogram('srcdata', encoder_input)
    tgtdata = tf.summary.histogram('tgtdata', decoder_input)

    encoder_output = encoder(encoder_input)
    logits, preds = decoder(decoder_input, encoder_output)

    loss_params = {
            'label_smoothing':0.1,
    }

    labels = padded_tgt_text[:, 1:]
    is_target = tf.to_float(tf.not_equal(labels, 0))
    targets_num = tf.reduce_sum(is_target)
    acc = tf.reduce_sum(tf.to_float(tf.equal(preds, labels))*is_target)/ targets_num
    acc_stat = tf.summary.scalar('acc', acc)

    onehot_labels = tf.one_hot(labels, depth=text_database.target_vocab.vocab_size)

    smoothed_labels = ( (1-loss_params['label_smoothing'])*onehot_labels) +\
        (loss_params['label_smoothing']/text_database.target_vocab.vocab_size)
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=smoothed_labels)
    mean_loss = tf.reduce_sum(loss*is_target) / targets_num
    #opt_hparams={
    #    "optimizer": {
    #        "type": "AdamOptimizer",
    #        "kwargs": {
    #            "learning_rate": 0.0001,
    #            "beta1": 0.9,
    #            "beta2": 0.98,
    #            "epsilon": 1e-8,
    #        }
    #    }
    #}
    #train_op, global_step = opt.get_train_op(mean_loss, hparams=opt_hparams)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.98, epsilon=1e-8)
    train_op = optimizer.minimize(mean_loss, global_step=global_step)
    tf.summary.scalar('mean_loss', mean_loss)
    merged = tf.summary.merge_all()

    saver = tf.train.Saver()
    # We shall wrap these environment setup codes
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                #pred loss, not include BOS
                # for epoch in range(num_epochs)
                source, target, preds = sess.run(
                    [encoder_input, labels, train_op, global_step, mean_loss, acc, merged],
                    feed_dict={context.is_train(): False})
                print('step:{} source:{}\n target:{}\n'.format(step, source, target))
                # for var in tf.trainable_variables():
                #     print('name:{}\tshape:{}\ttype:{}\n'.format(var.name, var.shape, var.dtype))
        except tf.errors.OutOfRangeError:
            print('Done -- epoch limit reached')
        finally:
            coord.request_stop()
        coord.join(threads)

