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
import os
import codecs
from nltk.translate.bleu_score import corpus_bleu
from data_load import load_test_data, load_de_vocab, load_en_vocab

if __name__ == "__main__":
    ### Build data pipeline

    # Config data hyperparams. Hyperparams not configured will be automatically
    # filled with default values. For text database, default values are defined
    # in `texar.data.database.default_text_dataset_hparams()`.
    data_hparams = {
        "num_epochs": 20,
        "seed": 123,
        "batch_size":32,
        "source_dataset": {
            "files": ['data/translation/de-en/test_de_sentences.txt'],
            "vocab_file": 'data/translation/de-en/filter_de.vocab.txt',
            "processing":{
                "bos_token": "<S>",
                "eos_token": "</S>",
                }
        },
        "target_dataset": {
            "files": ['data/translation/de-en/test_en_sentences.txt'],
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

    X, Sources, Targets = load_test_data()
    x = tf.placeholder(tf.int64, shape=(data_hparams['batch_size'], extra_hparams['max_seq_length']))
    y = tf.placeholder(tf.int64, shape=(data_hparams['batch_size'], extra_hparams['max_seq_length']))
    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()

    encoder = TransformerEncoder(vocab_size=len(de2idx),
            hparams=extra_hparams)
    decoder = TransformerDecoder(vocab_size=len(en2idx),
            hparams=extra_hparams)


    encoder_output = encoder(x)
    decoder_inputs = tf.concat((tf.ones_like(y[:,:1])*2, y[:,:-1]), -1)
    logits, preds = decoder(decoder_inputs, encoder_output)

    loss_params = {
            'label_smoothing':0.1,
    }

    y_smoothed = mle_losses.label_smoothing(y, len(idx2en), loss_params['label_smoothing'])
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_smoothed)

    is_target = tf.to_float(tf.not_equal(y, 0))
    acc = tf.reduce_sum(tf.to_float(tf.equal(preds, y))*is_target)/ (tf.reduce_sum(is_target))
    mle_loss = tf.reduce_sum(loss*is_target)/(tf.reduce_sum(is_target))

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

    sv = tf.train.Supervisor()

    with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sv.saver.restore(sess, tf.train.latest_checkpoint('logdir'))
        print('restored!')

        mname = open('dutil_logdir/checkpoint', 'r').read().split('"')[1] # model name
        if not os.path.exists('results'): os.mkdir('results')
        with codecs.open("results/" + mname, "w", "utf-8") as fout:
            list_of_refs, hypotheses = [], []

            for i in range(len(X) // data_hparams['batch_size']):
                src = X[i*data_hparams['batch_size']: (i+1)*data_hparams['batch_size']]
                sources = Sources[i*data_hparams['batch_size']: (i+1)*data_hparams['batch_size']]
                targets = Targets[i*data_hparams['batch_size']: (i+1)*data_hparams['batch_size']]
                outputs = np.zeros((data_hparams['batch_size'], extra_hparams['max_seq_length']), np.int32)
                for j in range(extra_hparams['max_seq_length']):
                    _preds = sess.run(preds, feed_dict={x: src, y: outputs})
                    outputs[:, j] = _preds[:, j]
                for source, target, pred in zip(sources, targets, outputs): # sentence-wise
                    got = " ".join(idx2en[idx] for idx in pred).split("</S>")[0].strip()
                    fout.write("- source: " + source +"\n")
                    fout.write("- expected: " + target + "\n")
                    fout.write("- got: " + got + "\n\n")
                    fout.flush()


                    ref = target.split()
                    hypothesis = got.split()
                    if len(ref) > 3 and len(hypothesis) > 3:
                        list_of_refs.append([ref])
                        hypotheses.append(hypothesis)
            score = corpus_bleu(list_of_refs, hypotheses)
            fout.write("Bleu Score = " + str(100*score))
        print('Done')

