#
""" Example for HRED structure.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name, no-name-in-module

import os
import numpy as np
import tensorflow as tf
import texar as tx

from texar.modules.encoders.hierarchical_encoders import HierarchicalRNNEncoder
from texar.modules.decoders.beam_search_decode import beam_search_decode

from tensorflow.contrib.seq2seq import tile_batch

from argparse import ArgumentParser

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from sw_loader import download_and_process
from config_data import data_root, max_utterance_cnt, data_hparams
download_and_process(data_root)

import importlib

flags = tf.flags
flags.DEFINE_string('config_model', 'config_model_biminor', 'The model config')
FLAGS = flags.FLAGS
config_model = importlib.import_module(FLAGS.config_model)

encoder_hparams = config_model.encoder_hparams
decoder_hparams = config_model.decoder_hparams
opt_hparams = config_model.opt_hparams

def main():
    # model part: data
    train_data = tx.data.MultiAlignedData(data_hparams['train'])
    val_data = tx.data.MultiAlignedData(data_hparams['val'])
    test_data = tx.data.MultiAlignedData(data_hparams['test'])
    iterator = tx.data.TrainTestDataIterator(train=train_data,
                                             val=val_data,
                                             test=test_data)
    data_batch = iterator.get_next()
    spk_src = tf.stack([data_batch['spk_{}'.format(i)] 
                        for i in range(max_utterance_cnt)], 1)
    spk_tgt = data_batch['spk_tgt']

    # model part: hred
    def add_source_speaker_token(x):
        return tf.concat([x, tf.reshape(spk_src, (-1, 1))], 1)

    def add_target_speaker_token(x):
        return (x, ) + (tf.reshape(spk_tgt, (-1, 1)), )

    embedder = tx.modules.WordEmbedder(
        init_value=train_data.embedding_init_value(0).word_vecs)
    encoder = HierarchicalRNNEncoder(hparams=encoder_hparams)

    decoder = tx.modules.BasicRNNDecoder(
        hparams=decoder_hparams, vocab_size=train_data.vocab(0).size)

    connector = tx.modules.connectors.MLPTransformConnector(
        decoder.cell.state_size)

    # build tf graph

    context_embed = embedder(data_batch['source_text_ids'])
    ecdr_states = encoder(
        context_embed,
        medium=['flatten', add_source_speaker_token],
        sequence_length=data_batch['source_length'],
        sequence_length_major=data_batch['source_utterance_cnt'])[1]

    ecdr_states = add_target_speaker_token(ecdr_states)
    dcdr_states = connector(ecdr_states)

    # train branch

    target_embed = embedder(data_batch['target_text_ids'])
    outputs, _, lengths = decoder(
        initial_state=dcdr_states,
        inputs=target_embed,
        sequence_length=data_batch['target_length'] - 1)

    mle_loss = tx.losses.sequence_sparse_softmax_cross_entropy(
        labels=data_batch['target_text_ids'][:, 1:],
        logits=outputs.logits,
        sequence_length=lengths,
        sum_over_timesteps=False,
        average_across_timesteps=True)

    global_step = tf.Variable(0, name='global_step', trainable=True)
    train_op = tx.core.get_train_op(
        mle_loss, global_step=global_step, hparams=opt_hparams)

    perplexity = tf.exp(mle_loss)

    # bleu branch
    dcdr_states_tiled = tile_batch(dcdr_states, 5)

    output_samples, _, sample_lengths = decoder(
        decoding_strategy="infer_sample",
        initial_state=dcdr_states_tiled,
        max_decoding_length=50,
        start_tokens=tf.cast(tf.fill(
            tf.shape(dcdr_states_tiled)[:1], train_data.vocab(0).bos_token_id),
            tf.int32),
        end_token=train_data.vocab(0).eos_token_id,
        embedding=embedder)

    # denumericalize the generated samples
    sample_text = train_data.vocab(0).map_ids_to_tokens(
        output_samples.sample_id)

    # beam search
    beam_search_samples, beam_states = beam_search_decode(
        decoder,
        initial_state=tile_batch(dcdr_states, 5),
        max_decoding_length=50,
        start_tokens=tf.cast(tf.fill(
            tf.shape(dcdr_states)[:1], train_data.vocab(0).bos_token_id),
            tf.int32),
        end_token=train_data.vocab(0).eos_token_id,
        embedding=embedder,
        beam_width=5)
    beam_lengths = beam_states.lengths

    beam_sample_text = train_data.vocab(0).map_ids_to_tokens(
        beam_search_samples.predicted_ids)

    target_tuple = (data_batch['target_text'][:, 1:],
                    data_batch['target_length'] - 1,
                    data_batch['target_text_ids'][:, 1:])
    #train_data.source_vocab.map_ids_to_tokens(
    #data_batch['target_text_ids'][:, 1:]),
    #data_batch['target_length'] - 1)

    dialog_tuple = (data_batch['source_text'], data_batch['source_length'],
                    data_batch['source_utterance_cnt'])

    refs_tuple = (data_batch['refs_text'][:, :, 1:], data_batch['refs_length'],
                  data_batch['refs_text_ids'][:, :, 1:], data_batch['refs_utterance_cnt'])


    def _train_epochs(sess, epoch, display=1000):
        iterator.switch_to_train_data(sess)

        for i in range(3000):
            try:
                feed = {tx.global_mode(): tf.estimator.ModeKeys.TRAIN}
                step, loss, _ = sess.run(
                    [global_step, mle_loss, train_op], feed_dict=feed)

                if step % display == 0:
                    print('step {} at epoch {}: loss={}'.format(
                        step, epoch, loss))

            except tf.errors.OutOfRangeError:
                break

        print('epoch {} train fin: loss={}'.format(epoch, loss))

    def _test_epochs_ppl(sess, epoch):
        iterator.switch_to_test_data(sess)

        pples = []
        while True:
            try:
                feed = {tx.global_mode(): tf.estimator.ModeKeys.EVAL}
                ppl = sess.run(perplexity, feed_dict=feed)
                pples.append(loss)

            except tf.errors.OutOfRangeError:
                avg_ppl = np.mean(pples)
                print('epoch {} perplexity={}'.format(epoch, avg_ppl))
                break

    def _test_epochs_bleu(sess, epoch):
        iterator.switch_to_test_data(sess)

        max_bleus = [[] for i in range(5)]
        avg_bleus = [[] for i in range(5)]

        def BLEU(hyps, refs, weight):
            prec = np.mean([max([sentence_bleu([ref], hyp, 
                smoothing_function=SmoothingFunction().method7, 
                weights=weights) for ref in refs]) for hyp in hyps])
            recall = np.mean([max([sentence_bleu([ref], hyp, 
                smoothing_function=SmoothingFunction().method7, 
                weights=weights) for hyp in hyps]) for ref in refs])

            return prec, recall

        while batch_cnt != test_batch_num:
            try:
                feed = {tx.global_mode(): tf.estimator.ModeKeys.EVAL}

                samples, sample_id, lengths, dialog_t, target_t, refs_t = sess.run(
                    [sample_text, output_samples.sample_id, sample_lengths,
                     dialog_tuple, target_tuple, refs_tuple],
                    feed_dict=feed)

                samples = samples.reshape(-1, 5, *samples.shape[1:]).transpose(0, 2, 1)
                sample_id = sample_id.reshape(-1, 5, *sample_id.shape[1:]).transpose(0, 2, 1)
                lengths = lengths.reshape(-1, 5, *lengths.shape[1:])

                for (beam, beam_len, beam_ids,
                     dialog, utts_len, utts_cnt,
                     target, tgt_len, tgt_ids,
                     refs, refs_len, refs_ids, refs_cnt) in zip(
                    samples, lengths, sample_id, *dialog_t, *target_t, *refs_t):

                    srcs = [dialog[i, :utts_len[i]] for i in range(utts_cnt)]
                    hyps = [beam[:l-1, i] for i, l in enumerate(beam_len)]
                    hyps_ids = [beam_ids[:l-1, i] for i, l in enumerate(beam_len)]
                    refs = [refs[i, :refs_len[i]-1] for i in range(refs_cnt)][:6]
                    refs += [target[:tgt_len-1]]
                    refs_ids = [refs_ids[i, :refs_len[i]-1] for i, l in enumerate(range(refs_cnt))][:6]
                    refs_ids += [tgt_ids[:tgt_len-1]]

                    hyps_ids = [hyp_ids for hyp_ids in hyps_ids if len(hyp_ids) > 0]

                    embedding = test_data.embedding_init_value(0).word_vecs

                    for bleu_i in range(1, 5):
                        weights = [1. / bleu_i, ] * bleu_i

                        scrs = []

                        for hyp in hyps:
                            try:    
                                scrs.append(sentence_bleu(refs, hyp,
                                    smoothing_function=SmoothingFunction().method7,
                                    weights=weights))
                            except:
                                pass
                                #scrs.append(0)

                        if len(scrs) == 0:
                            scrs.append(0)

                        max_bleu, avg_bleu = np.max(scrs), np.mean(scrs)
                        max_bleus[bleu_i].append(max_bleu)
                        avg_bleus[bleu_i].append(avg_bleu)

            except tf.errors.OutOfRangeError:
                break

        bleu_recall = [np.mean(max_bleus[i]) for i in range(1, 5)]
        bleu_prec = [np.mean(avg_bleus[i]) for i in range(1, 5)]

        for i in range(1, 5):
            print('BLEU-{} prec={}, BLEU-{} recall={}'.format(
                i, bleu_prec[i-1], i, bleu_recall[i-1])) 

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for epoch in range(10):
            _train_epochs(sess, epoch)
            _test_epochs_ppl(sess, epoch)
            _test_epochs_bleu(sess, epoch)

if __name__ == "__main__":
    main()
