# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''

from __future__ import print_function
import codecs

import tensorflow as tf
import numpy as np

from data_load import load_test_data, load_de_vocab, load_en_vocab, hp
from nltk.translate.bleu_score import corpus_bleu
from texar.modules import TransformerEncoder, TransformerDecoder
from texar.losses import mle_losses
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
                'type':tf.contrib.layers.xavier_initializer(),
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
    test_corpus, source_list, target_list = load_test_data()
    src_input = tf.placeholder(tf.int32, shape=(hp.batch_size, \
        hp.maxlen))
    tgt_input = tf.placeholder(tf.int32, shape=(hp.batch_size, \
        hp.maxlen))
    src_length = tf.reduce_sum(tf.to_float(tf.not_equal(src_input, 0)), axis=-1)

    de2idx, _ = load_de_vocab()
    en2idx, idx2en = load_en_vocab()
    decoder_input = tf.concat((tf.ones_like(tgt_input[:, :1]), tgt_input[:, :-1]), -1) # 1:<S>
    tgt_length = tf.reduce_sum(tf.to_float(tf.not_equal(decoder_input, 0)), axis=-1)

    encoder = TransformerEncoder(vocab_size=len(de2idx), hparams=extra_hparams)
    encoder_output = encoder(src_input,
        inputs_length=src_length)

    decoder = TransformerDecoder(vocab_size=len(en2idx), hparams=extra_hparams)
    logits, preds = decoder(decoder_input,
        encoder_output,
        src_length=src_length,
        tgt_length=tgt_length)

    loss_params = {
        'label_smoothing':0.1,
    }
    is_target=tf.to_float(tf.not_equal(tgt_input, 0))
    smoothed_labels = mle_losses.label_smoothing(tgt_input, len(idx2en), loss_params['label_smoothing'])
    mle_loss = mle_losses.average_sequence_softmax_cross_entropy(
        labels=smoothed_labels,
        logits=logits,
        sequence_length=tf.reduce_sum(is_target, -1))

    # Start session
    #print('var cnt:{}'.format(len(tf.trainable_variables())))
    #for var in tf.trainable_variables():
    #    print('var: name:{} shape:{} dtype:{}'.format(var.name, var.shape, var.dtype))

    #model_path = './logdir/'
    model_path = './dutil_logdir/'
    #model_path = '../transformer/logdir/BLEU17.66/'

    with tf.Session() as sess:
        ## Restore parameters
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())
        """
            Here we also provide an interface to load the trained model of
                https://github.com/Kyubyong/transformer
            What we have done here is just variabla-name matching
        """
        if model_path.startswith('../transformer/logdir'):
            varlist = tf.trainable_variables()
            namelist = [ var.name for var in varlist]
            namelist = [name.replace('encoder_1','encoder') for name in namelist]
            namelist = [name.replace('decoder_1','decoder') for name in namelist]
            namelist = [name.replace('decoder/dense/', 'dense/') for name in namelist]
            namelist = [name.replace('encoder/lookup_table','encoder/enc_embed/lookup_table') for name in namelist]
            namelist = [name.replace('decoder/lookup_table','decoder/dec_embed/lookup_table') for name in namelist]
            namelist = [name[:-2] if name[-2]==':' else name for name in namelist]
            for i in range(6):
                namelist = [name.replace('decoder/num_blocks_{}/multihead_attention_1'.format(i), \
                    'decoder/num_blocks_{}/multihead_attention'.format(i)) for name in namelist]
                namelist = [name.replace('encoder/num_blocks_{}/multihead_attention_1/'.format(i),
                    'encoder/num_blocks_{}/multihead_attention/'.format(i)) for name in namelist]
                namelist = [name.replace('encoder/num_blocks_{}/multihead_attention_1_1/ln'.format(i), \
                    'encoder/num_blocks_{}/multihead_attention_1/ln'.format(i)) for name in namelist]
            vardict={}
            for var, name in zip(varlist, namelist):
                vardict[name]=var
            saver = tf.train.Saver(vardict)
            saver.restore(sess, tf.train.latest_checkpoint(model_path))

        elif model_path =='./logdir/' or model_path.startswith('./dutil_logdir/'):
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(model_path))
        mname = tf.train.latest_checkpoint(model_path).split('/')[-1]
        print('model name:{}'.format(mname))
        ## Inference
        resultfile = model_path + mname +'.result.txt'
        print('result:{}'.format(resultfile))
        with codecs.open(resultfile, "w", "utf-8") as fout:
            list_of_refs, hypotheses = [], []
            for i in range(len(test_corpus) // hp.batch_size):
                src = test_corpus[i*hp.batch_size: (i+1)*hp.batch_size]
                sources = source_list[i*hp.batch_size: (i+1)*hp.batch_size]
                targets = target_list[i*hp.batch_size: (i+1)*hp.batch_size]

                outputs = np.zeros((hp.batch_size, hp.maxlen),\
                    np.int32)
                for j in range(hp.maxlen):
                    loss, _preds = sess.run([mle_loss, preds], \
                        feed_dict={
                            src_input: src, tgt_input: outputs,
                            context.global_mode(): tf.estimator.ModeKeys.EVAL})
                    #fout.write('loss:{}\n'.format(loss))
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
            print("Bleu Score = " + str(100*score))

if __name__ == '__main__':
    evaluate()
    print("Done")

