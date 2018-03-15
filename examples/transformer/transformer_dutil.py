"""
Example pipeline. This is a minimal example of basic RNN language model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name, no-name-in-module
import random
import numpy as np
import tensorflow as tf
from texar.modules import TransformerEncoder, TransformerDecoder
from texar.losses import mle_losses
from texar.core import optimization as opt
from texar import context

from data_load import get_batch_data, load_de_vocab, load_en_vocab

if __name__ == "__main__":
    tf.set_random_seed(123)
    np.random.seed(123)
    random.seed(123)
    extra_hparams = {
        'max_seq_length':10,
        'scale':True,
        'sinusoid':False,
        'embedding': {
            'name': 'lookup_table',
            'dim': 512,
            'initializer': {
                'type': tf.contrib.layers.xavier_initializer(),
            },
            'trainable':True,
        },
        'num_blocks': 6,
        'num_heads': 8,
        'poswise_feedforward': {
            'name':'multihead_attention',
            'layers':[
                {
                    'type':'Conv1D',
                    'kwargs': {
                        'filters':512*4,
                        'kernel_size':1,
                        'activation':'relu',
                        'use_bias':True,
                    }
                },
                {
                    'type':'Conv1D',
                    'kwargs': {
                        'filters':512,
                        'kernel_size':1,
                        'use_bias':True,
                    }
                }
            ],
        },
    }
    src_input, tgt_input, num_batch = get_batch_data()
    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()

    decoder_input = tf.concat((tf.ones_like(tgt_input[:, :1]), tgt_input[:, :-1]), -1)

    encoder = TransformerEncoder(vocab_size=len(de2idx), hparams=extra_hparams)
    encoder_output = encoder(src_input)

    decoder = TransformerDecoder(vocab_size=len(en2idx), hparams=extra_hparams)
    logits, preds = decoder(decoder_input, encoder_output)
    loss_params = {
        'label_smoothing':0.1,
    }
    istarget = tf.to_float(tf.not_equal(tgt_input, 0))
    smoothed_labels = mle_losses.label_smoothing(tgt_input, len(en2idx), loss_params['label_smoothing'])

    mle_loss = mle_losses.average_sequence_softmax_cross_entropy(
        labels=smoothed_labels,
        logits=logits,
        sequence_length=tf.reduce_sum(istarget, -1))
    tf.summary.scalar('mle_loss', mle_loss)
    acc = tf.reduce_sum(tf.to_float(tf.equal(preds, tgt_input))*istarget) / (tf.reduce_sum(istarget))
    tf.summary.scalar('acc', acc)
    opt_hparams = {
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
    #global_step = tf.Variable(0, name='global_step', trainable=False)
    #optimizer = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.98, epsilon=1e-8)
    #train_op = optimizer.minimize(mle_loss, global_step=global_step)
    merged = tf.summary.merge_all()

    #print('var cnt:{}'.format(len(tf.trainable_variables())))
    #for var in tf.trainable_variables():
    #    print('name:{}\tshape:{}\ttype:{}'.format(var.name, var.shape, var.dtype))
    logdir = './dutil_logdir/'
    # We shall wrap these environment setup codes
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())
        saver = tf.train.Saver(max_to_keep=10)
        writer = tf.summary.FileWriter(logdir, graph=sess.graph)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            while not coord.should_stop():
                for epoch in range(20):
                    for batch_idx in range(num_batch):
                        source, target, predict, _, gstep, loss, mgd = sess.run(
                            [src_input, tgt_input, preds, train_op, global_step, mle_loss, merged],
                            feed_dict={context.is_train(): True})
                    writer.add_summary(mgd, global_step=gstep)
                    assert gstep % 1703 == 0
                    print('step:{} loss:{}'.format(gstep, loss))
                    saver.save(sess, logdir+'my-model', global_step=gstep)
                coord.request_stop()
        except tf.errors.OutOfRangeError:
            print('Done -- epoch limit reached')
        finally:
            coord.request_stop()
        coord.join(threads)
        saver.save(sess, logdir+'my-model', global_step=gstep)
