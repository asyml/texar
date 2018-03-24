# -*- coding: utf-8 -*-
#/usr/bin/python2
from __future__ import print_function
import codecs
import beam_utils
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
        None))
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
    likelihoods = tf.nn.log_softmax(logits, dim=-1)
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
        with codecs.open(resultfile, "w", "utf-8") as fout, codecs.open(outputfile, 'w+','utf-8') as oout:
            list_of_refs, hypotheses = [], []
            for i in range(len(test_corpus) // hp.batch_size):
                print('i:{}'.format(i))
                assert hp.batch_size == 1, 'batch beam search is not supported'
                active_hyp = [beam_utils.Hypothesis(0, [], None)]
                completed_hyp = []
                length = 0
                src = test_corpus[i*hp.batch_size: (i+1)*hp.batch_size]
                sources = source_list[i*hp.batch_size: (i+1)*hp.batch_size]
                targets = target_list[i*hp.batch_size: (i+1)*hp.batch_size]
                #outputs = np.ones((hp.batch_size, hp.maxlen), np.int32) #这里应该从BOS开始
                while len(completed_hyp) < hp.beam_size and length < hp.maxlen:
                    new_set = []
                    for hyp in active_hyp:
                        if length > 0 :
                            if hyp.output[-1] == 2: #EOS
                                hyp.output = hyp.output[:-1]
                                completed_hyp.append(hyp)
                                continue
                        tmp_tgt_input = [ [1] + hyp.output]
                        _, _scores, _preds = sess.run([mle_loss, likelihoods[0], preds[0]], \
                            feed_dict={
                                src_input: src,
                                tgt_input: tmp_tgt_input,
                                context.is_train():False,
                            })
                        _scores = _scores[length]
                        #print(_scores.shape)
                        top_ids = np.argpartition(_scores, max(-len(_scores), -2*hp.beam_size))[-2*hp.beam_size:]
                        for cur_id in top_ids.tolist():
                            new_list = list(hyp.output)
                            new_list.append(cur_id)
                            log_score = beam_utils.default_len_normalize_partial(hyp.score, _scores[cur_id], len(new_list))
                            new_set.append(beam_utils.Hypothesis(log_score, new_list, None))
                    length += 1
                    active_hyp = sorted(new_set, key=lambda x: x.score, reverse=True)[:hp.beam_size]
                if len(completed_hyp) == 0:
                    completed_hyp = active_hyp
                #beam_utils.normalize_completed(completed_hyp, x_length)
                results = [sorted(completed_hyp, key=lambda x:x.score, reverse=True)[0]]
                for source, target, pred in zip(sources, targets, results): # sentence-wise
                    pred = pred.output
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
