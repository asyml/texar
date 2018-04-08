# -*- coding: utf-8 -*-
#/usr/bin/python2
from __future__ import print_function
import codecs
import tensorflow as tf
#import numpy as np
import texar
from texar.data import qPairedTextData
from hyperparams import args as hp
from nltk.translate.bleu_score import corpus_bleu
from texar.modules import TransformerEncoder, TransformerDecoder
from texar import context
from hyperparams import encoder_hparams, decoder_hparams, test_dataset_hparams

def evaluate():
    print("Graph loaded")
    # Load data
    test_database = qPairedTextData(test_dataset_hparams)
    vocab = test_database.target_vocab
    text_data_batch = test_database()
    ori_src_text = text_data_batch['source_text_ids']
    ori_tgt_text = text_data_batch['target_text_ids']
    encoder_input = ori_src_text[:, 1:]
    enc_input_length = tf.reduce_sum(tf.to_float(tf.not_equal(encoder_input, 0)))
    WordEmbedder = texar.modules.WordEmbedder(
        vocab_size=test_database.source_vocab.size,
        hparams=hp.word_embedding_hparams,
    )
    encoder = TransformerEncoder(vocab_size=test_database.source_vocab.size, \
        embedding=WordEmbedder._embedding,
        hparams=encoder_hparams)
    encoder_output, encoder_decoder_attention_bias = encoder(encoder_input,
        inputs_length=enc_input_length)
    decoder = TransformerDecoder(
        embedding = encoder._embedding,
        hparams=decoder_hparams
    )
    #print('var cnt:{}'.format(len(tf.trainable_variables())))
    #for var in tf.trainable_variables():
    #    print('var: name:{} shape:{} dtype:{}'.format(var.name, var.shape, var.dtype))

    model_path = hp.log_dir
    #model_path = './dutil_logdir/'
    #model_path = '../transformer/logdir/BLEU17.66/'
    predictions = decoder.dynamic_decode(encoder_output,
                                        encoder_decoder_attention_bias)
    with tf.Session() as sess:
        ## Restore parameters
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())
        saver = tf.train.Saver()
        print('model path:{}'.format(model_path))
        #var_map = {}
        #for var in tf.trainable_variables():
        #    if var.name.startswith('lookup_table'):
        #        var_map['encoder/lookup_table'] = var
        #    elif var.name.endswith(':0'):
        #        var_map[var.name[:-2]] = var
        #    else:
        #        var_map[var.name] = var
        #saver = tf.train.Saver(var_list=var_map)
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
        mname = tf.train.latest_checkpoint(model_path).split('/')[-1]
        print('model name:{}'.format(mname))
        ## Inference
        resultfile = model_path + mname + '.b{}alpha{}.result.txt'.format(hp.beam_width, hp.alpha)
        outputfile = model_path + mname + '.b{}alpha{}.output.txt'.format(hp.beam_width, hp.alpha)
        print('result:{}'.format(resultfile))
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        list_of_refs, hypotheses = [], []
        with codecs.open(resultfile, "w", "utf-8") as fout, codecs.open(outputfile, 'w+','utf-8') as oout:
            try:
                while not coord.should_stop():
                    #outputs = np.ones((hp.batch_size, hp.maxlen), np.int32) #这里应该从BOS开始
                    fetches = [predictions, ori_src_text, ori_tgt_text]
                    predicts, src, tgt = sess.run(fetches, \
                        feed_dict={
                            context.global_mode(): tf.estimator.ModeKeys.PREDICT,
                        }
                    )
                    sources, sampled_ids, targets = src.tolist(), predicts['sampled_ids'], tgt[:, 1:].tolist()

                    #print('sampled_ids:{}'.format(sampled_ids.shape))
                    #[batch_Size, top_k, token_ids]
                    sampled_ids = sampled_ids[:, 0, :]
                    sources = [' '.join([vocab._id_to_token_map_py[i] for i in sent]) for sent in sources]
                    targets = [' '.join([vocab._id_to_token_map_py[i] for i in sent]) for sent in targets]
                    dwords = [ ' '.join([vocab._id_to_token_map_py[i] for i in sent]) for sent in sampled_ids]
                    print('the {}-th sentence'.format(len(list_of_refs)))
                    for source, target, pred in zip(sources, targets, dwords): # sentence-wise
                        source = source.split('<EOS>')[0].strip()
                        target = target.split('<EOS>')[0].strip()
                        got = pred.split("<EOS>")[0].strip()
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
            except tf.errors.OutOfRangeError:
                print('Done -- epoch limit reached')
            finally:
                coord.request_stop()
            coord.join(threads)
            ## Calculate bleu score
            score = corpus_bleu(list_of_refs, hypotheses)
            fout.write("Bleu Score = " + str(100*score))
            print("Bleu Score = " + str(100*score))

if __name__ == '__main__':
    evaluate()
    print("Done")
