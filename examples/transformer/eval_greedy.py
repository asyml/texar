# -*- coding: utf-8 -*-
#/usr/bin/python2
from __future__ import print_function
import codecs

import tensorflow as tf
import numpy as np
from data_load import load_test_data, load_shared_vocab, hp
from nltk.translate.bleu_score import corpus_bleu
from texar.modules import TransformerEncoder, TransformerDecoder
from texar.losses import mle_losses
from texar import context
from hyperparams import encoder_hparams, decoder_hparams

def evaluate():
    print("Graph loaded")
    # Load data
    test_corpus, source_list, target_list = load_test_data()
    src_input = tf.placeholder(tf.int32, shape=(hp.batch_size, \
        hp.maxlen))
    tgt_input = tf.placeholder(tf.int32, shape=(hp.batch_size, \
        hp.maxlen))
    src_length = tf.reduce_sum(tf.to_float(tf.not_equal(src_input, 0)), axis=-1)

    word2idx, idx2word = load_shared_vocab()
    decoder_input = tf.concat((tf.ones_like(tgt_input[:, :1]), tgt_input[:, :-1]), -1) # 1:<S>
    tgt_length = tf.reduce_sum(tf.to_float(tf.not_equal(decoder_input, 0)), axis=-1)

    encoder = TransformerEncoder(vocab_size=len(word2idx), hparams=encoder_hparams)
    encoder_output = encoder(src_input,
        inputs_length=src_length)

    decoder = TransformerDecoder(
        embedding = encoder._embedding,
        hparams=decoder_hparams)
    logits, preds = decoder(
        decoder_input,
        encoder_output,
        src_length=src_length,
        tgt_length=tgt_length)
    loss_params = {
        'label_smoothing':0.1,
    }
    is_target=tf.to_float(tf.not_equal(tgt_input, 0))
    smoothed_labels = mle_losses.label_smoothing(tgt_input, len(idx2word), loss_params['label_smoothing'])
    mle_loss = mle_losses.average_sequence_softmax_cross_entropy(
        labels=smoothed_labels,
        logits=logits,
        sequence_length=tf.reduce_sum(is_target, -1))

    # Start session
    #print('var cnt:{}'.format(len(tf.trainable_variables())))
    #for var in tf.trainable_variables():
    #    print('var: name:{} shape:{} dtype:{}'.format(var.name, var.shape, var.dtype))

    model_path = './logdir/'
    #model_path = './dutil_logdir/'
    #model_path = '../transformer/logdir/BLEU17.66/'

    with tf.Session() as sess:
        ## Restore parameters
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
        mname = tf.train.latest_checkpoint(model_path).split('/')[-1]
        print('model name:{}'.format(mname))
        ## Inference
        resultfile = model_path + mname + '.result.txt'
        outputfile = model_path + mname + '.output.txt'
        print('result:{}'.format(resultfile))
        print('test corpus size:{} source_list:{}'.format(len(test_corpus), len(source_list)))
        with open('extracted_test.txt', 'w+') as outfile:
            for sent in source_list:
                outfile.write(sent+'\n')
        overall_sources, overall_targets, overall_outputs = [], [], []
        with codecs.open(resultfile, "w", "utf-8") as fout, codecs.open(outputfile, 'w+','utf-8') as oout:
            list_of_refs, hypotheses = [], []
            for i in range(len(test_corpus) // hp.batch_size):
                print('i:{} instance:{}'.format(i, i*hp.batch_size))
                src = test_corpus[i*hp.batch_size: (i+1)*hp.batch_size]
                sources = source_list[i*hp.batch_size: (i+1)*hp.batch_size]
                targets = target_list[i*hp.batch_size: (i+1)*hp.batch_size]
                outputs = np.zeros((hp.batch_size, hp.maxlen),np.int32)
                finished = [False] * hp.batch_size
                for j in range(hp.maxlen):
                    _, _preds = sess.run([mle_loss, preds], \
                        feed_dict={
                            src_input: src,
                            tgt_input: outputs,
                            context.is_train():False
                        })
                    for k in range(hp.batch_size):
                        if _preds[k][j] == word2idx['<EOS>']:
                            finished[k] = True
                    outputs[:, j] = _preds[:, j]
                    if False not in finished:
                        break
                overall_sources.extend(sources)
                overall_targets.extend(targets)
                overall_outputs.extend(outputs)

            for source, target, pred in zip(overall_sources, overall_targets, overall_outputs): # sentence-wise
                got = " ".join(idx2word[idx] for idx in pred).split("<EOS>")[0].strip()
                fout.write("- source: " + source +"\n")
                fout.write("- expected: " + target + "\n")
                fout.write("- got: " + got + "\n\n")
                fout.flush()
                oout.write(got+'\n')
                # bleu score
                ref = target.split()
                hypothesis = got.split()
                list_of_refs.append([ref])
                hypotheses.append(hypothesis)

            ## Calculate bleu score
            score = corpus_bleu(list_of_refs, hypotheses)
            fout.write("Bleu Score = " + str(100*score))
            print("Bleu Score = " + str(100*score))

if __name__ == '__main__':
    evaluate()
    print("Done")
