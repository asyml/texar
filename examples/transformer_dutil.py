"""
Example pipeline. This is a minimal example of basic RNN language model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name, no-name-in-module

import tensorflow as tf

# We shall wrap all these modules
from txtgen.modules import TransformerEncoder, TransformerDecoder
from txtgen.losses import mle_losses
from txtgen.core import optimization as opt
from txtgen import context
import time
import numpy as np
import random
from data_load import get_batch_data, load_de_vocab, load_en_vocab
random.seed(123)
np.random.seed(123)
tf.set_random_seed(123)

if __name__ == "__main__":
    ### Build data pipeline

    # Config data hyperparams. Hyperparams not configured will be automatically
    # filled with default values. For text database, default values are defined
    # in `txtgen.data.database.default_text_dataset_hparams()`.
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
    x, y, num_batch= get_batch_data()
    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()

    encoder = TransformerEncoder(vocab_size=len(de2idx),
            hparams=extra_hparams)
    decoder = TransformerDecoder(vocab_size=len(idx2en),
            hparams=extra_hparams)

    encoder_output = encoder(x)
    decoder_inputs = tf.concat((tf.ones_like(y[:,:1])*2, y[:,:-1]), -1)

    logits, preds = decoder(decoder_inputs, encoder_output)

    loss_params = {
            'label_smoothing':0.1,
    }

    is_target=tf.to_float(tf.not_equal(y, 0))
    targets_num = tf.reduce_sum(is_target)
    correct_num = tf.reduce_sum(tf.to_float(tf.equal(preds, y))*is_target)
    acc = correct_num / targets_num

    y_smoothed = mle_losses.label_smoothing(y, len(idx2en), loss_params['label_smoothing'])
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_smoothed)
    mle_loss = tf.reduce_sum(loss*is_target)/(tf.reduce_sum(is_target))

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
    sv = tf.train.Supervisor(graph=tf.get_default_graph(),
            logdir='dutil_logdir',
            save_model_secs=0,
            # init_feed_dict={context.is_train():True},
            )

    # We shall wrap these environment setup codes
    with sv.managed_session() as sess:
        with open('dutil_logdir/datafeed_tgt.txt', 'w+') as outfile:
            while not sv.should_stop():
                source, target, _, step, loss, accuracy, cnum, tnum= sess.run(
                    [x, y, train_op, global_step, mle_loss, acc, correct_num, targets_num],
                )
                # for var in tf.trainable_variables():
                #     print('name:{}\tshape:{}\ttype:{}\n'.format(var.name, var.shape, var.dtype))
                # exit()
                outfile.write('target:{}\n'.format('\n'.join([' '.join([idx2en[i] for i in line]) \
                        for line in target])))
                if step % 100 == 1:
                    print("time{} step{} loss{} acc{}={}/{}".format(time.asctime(time.localtime(time.time())), step,\
                        loss, accuracy, cnum, tnum))
                if step % 2000 == 0:
                    sv.saver.save(sess, 'dutil_logdir/dutil_model', global_step = step)
