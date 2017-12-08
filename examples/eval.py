# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''

from __future__ import print_function
import codecs
import os

import tensorflow as tf
import numpy as np

from data_load import load_test_data, load_de_vocab, load_en_vocab, data_hparams
from nltk.translate.bleu_score import corpus_bleu
from texar.modules import TransformerEncoder, TransformerDecoder
def eval():
    # Load graph

    print("Graph loaded")

    # Load data
    extra_hparams = {
            'max_seq_length':10,
            'scale':True,
            'sinusoid':False,
            'embedding': {
                'name':'lookup_table',
                'initializer': {
                    'type':'xavier_initializer',
                    },
                'dim': 512,
                },
            'num_blocks': 6,
            'num_heads': 8,
    }
    X, Sources, Targets = load_test_data()
    x = tf.placeholder(tf.int32, shape=(data_hparams['batch_size'], data_hparams['max_seq_length']))
    y = tf.placeholder(tf.int32, shape=(data_hparams['batch_size'], data_hparams['max_seq_length']))
    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()
    encoder = TransformerEncoder(vocab_size=len(de2idx),
            hparams=extra_hparams)
    decoder = TransformerDecoder(vocab_size=len(idx2en),
            hparams=extra_hparams)
    encoder_output = encoder(x)
    logits, preds = decoder(y, encoder_output)

    # Start session
    sv = tf.train.Supervisor(
            saver = tf.train.Saver(tf.trainable_variables()),
            save_model_secs=0)
    with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        ## Restore parameters
        sv.saver.restore(sess, tf.train.latest_checkpoint('../transformer/logdir'))
        #print('var cnt:{}'.format(len(tf.trainable_variables())))
        #for var in tf.trainable_variables():
        #    print('var name:{} shape:{} dtype:{}'.format(var.name, var.shape, var.dtype))
        #exit()
        print("Restored!")

        ## Get model name
        mname = open('../transformer/logdir/checkpoint', 'r').read().split('"')[1] # model name
        print('mname:{}'.format(mname))
        ## Inference
        if not os.path.exists('results'): os.mkdir('results')
        with codecs.open("results/" + mname, "w", "utf-8") as fout:
            list_of_refs, hypotheses = [], []
            for i in range(len(X) // data_hparams['batch_size']):
                ### Get mini-batches
                src = X[i*data_hparams['batch_size']: (i+1)*data_hparams['batch_size']]
                sources = Sources[i*data_hparams['batch_size']: (i+1)*data_hparams['batch_size']]
                targets = Targets[i*data_hparams['batch_size']: (i+1)*data_hparams['batch_size']]
                ### Autoregressive inference
                outputs = np.zeros((data_hparams['batch_size'], data_hparams['max_seq_length']), np.int32)
                for j in range(data_hparams['max_seq_length']):
                    print('begin fetch')
                    _preds = sess.run(preds, feed_dict={x: src, y: outputs})
                    print('run over')
                    outputs[:, j] = _preds[:, j]

                ### Write to file
                for source, target, pred in zip(sources, targets, outputs): # sentence-wise
                    got = " ".join(idx2en[idx] for idx in pred).split("</S>")[0].strip()
                    fout.write("- source: " + source +"\n")
                    fout.write("- expected: " + target + "\n")
                    fout.write("- got: " + got + "\n\n")
                    fout.flush()

                    # bleu score
                    ref = target.split()
                    hypothesis = got.split()
                    if len(ref) > 3 and len(hypothesis) > 3:
                        list_of_refs.append([ref])
                        hypotheses.append(hypothesis)

            ## Calculate bleu score
            score = corpus_bleu(list_of_refs, hypotheses)
            fout.write("Bleu Score = " + str(100*score))

if __name__ == '__main__':
    eval()
    print("Done")


