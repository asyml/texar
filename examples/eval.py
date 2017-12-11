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
from texar import context
def evaluate():
    """
    evaluate the saved model in logdir directory
    """
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
    test_corpus, sources, targets = load_test_data()
    src_input = tf.placeholder(tf.int32, shape=(data_hparams['batch_size'], \
        data_hparams['max_seq_length']))
    tgt_input = tf.placeholder(tf.int32, shape=(data_hparams['batch_size'], \
        data_hparams['max_seq_length']))
    de2idx, _ = load_de_vocab()
    en2idx, idx2en = load_en_vocab()
    decoder_input = tf.concat((tf.ones_like(tgt_input[:, :1]), tgt_input[:, :-1]), -1) # 1:<S>

    encoder = TransformerEncoder(vocab_size=len(de2idx), hparams=extra_hparams)
    encoder_output = encoder(src_input)

    decoder = TransformerDecoder(vocab_size=len(en2idx), hparams=extra_hparams)

    #vocab=text_database.target_vocab
    _, preds = decoder(decoder_input, encoder_output)

    # Start session
    #print('var cnt:{}'.format(len(tf.trainable_variables())))
    #for var in tf.trainable_variables():
    #    print('var: name:{} shape:{} dtype:{}'.format(var.name, var.shape, var.dtype))

    with tf.Session() as sess:
        ## Restore parameters
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())
        saver = tf.train.Saver(tf.trainable_variables())
        #for var in tf.trainable_variables():
        #    print('var name:{} shape:{} dtype:{}'.format(var.name, var.shape, var.dtype))
        #exit()


        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint('./logdir'))
        #writer = tf.summary.FileWriter('eval/', sess.graph)
        #exit()
        ## Get model name
        mname = open('./logdir/checkpoint', 'r').read().split('"')[1] # model name
        print('mname:{}'.format(mname))
        ## Inference
        if not os.path.exists('results'):
            os.mkdir('results')
        with codecs.open("results/" + mname, "w", "utf-8") as fout:
            list_of_refs, hypotheses = [], []
            for i in range(len(test_corpus) // data_hparams['batch_size']):
                src = test_corpus[i*data_hparams['batch_size']: (i+1)*data_hparams['batch_size']]
                sources = sources[i*data_hparams['batch_size']: (i+1)*data_hparams['batch_size']]
                targets = targets[i*data_hparams['batch_size']: (i+1)*data_hparams['batch_size']]

                outputs = np.zeros((data_hparams['batch_size'], data_hparams['max_seq_length']),\
                    np.int32)
                for j in range(data_hparams['max_seq_length']):
                    _, _preds = sess.run([encoder.enc, preds], \
                        feed_dict={src_input: src, tgt_input: outputs, context.is_train():False})
                    outputs[:, j] = _preds[:, j]
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
    evaluate()
    print("Done")
