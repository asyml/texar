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

    decoder_input =tf.concat((tf.ones_like(y[:, :1]), y[:, :-1]), -1) # 1:<S>

    encoder= TransformerEncoder(vocab_size=len(de2idx),
            hparams=extra_hparams)
    encoder_output=encoder(x)

    decoder = TransformerDecoder(vocab_size=len(idx2en),
            hparams=extra_hparams)

    #vocab=text_database.target_vocab

    #helper_infer=get_helper(
    #        decoder.hparams.helper_infer.type,
    #        embedding=decoder.embedding,
    #        start_tokens=[vocab._token_to_id_map_py[vocab.bos_token]]*data_hparams['max_sequence_length'],
    #        end_token=vocab._token_to_id_map[vocab.eos_token],
    #        softmax_temperature=None)



    logits, preds = decoder(decoder_input, encoder_output)
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
        #sv.saver.restore(sess, tf.train.latest_checkpoint('./logdir'))
        #print('var cnt:{}'.format(len(tf.trainable_variables())))

        #for var in tf.trainable_variables():
        #    print('var name:{} shape:{} dtype:{}'.format(var.name, var.shape, var.dtype))
        #exit()

        varlist =tf.trainable_variables()
        namelist = [var.name for var in varlist]
        newnamelist = [name[:7]+'_1'+name[7:] if (name[7]=='/' and (name[8]=='n' or name[8]=='d' or name[8]=='e')) else name for name in namelist]
        newnamelist = [name[:-2] if name[-2]==':' else name for name in newnamelist]
        vardict={}
        for name, var in zip(newnamelist, varlist):
            vardict[name]=var
        saver = tf.train.Saver(vardict)
        saver.restore(sess, tf.train.latest_checkpoint('./logdir'))
        #writer = tf.summary.FileWriter('eval/', sess.graph)
        #exit()
        ## Get model name
        mname = open('./logdir/checkpoint', 'r').read().split('"')[1] # model name
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
                    _preds = sess.run(preds, feed_dict={x: src, y: outputs})
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


