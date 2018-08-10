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

from config_model import encoder_hparams, decoder_hparams, opt_hparams 

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

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for epoch in range(10):
            _train_epochs(sess, epoch)
            _test_epochs_ppl(sess, epoch)

if __name__ == "__main__":
    main()
