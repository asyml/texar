# Copyright 2018 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Hierarchical Recurrent Encoder-Decoder (HRED) for dialog response
generation.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name, too-many-locals

import importlib
import numpy as np
import tensorflow as tf
import texar as tx

from config_data import max_utterance_cnt, data_hparams

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

flags = tf.flags

flags.DEFINE_string('config_model', 'config_model_biminor', 'The model config')

FLAGS = flags.FLAGS

config_model = importlib.import_module(FLAGS.config_model)

encoder_hparams = config_model.encoder_hparams
decoder_hparams = config_model.decoder_hparams
opt_hparams = config_model.opt_hparams

def main():
    """Entrypoint.
    """
    # Data
    train_data = tx.data.MultiAlignedData(data_hparams['train'])
    val_data = tx.data.MultiAlignedData(data_hparams['val'])
    test_data = tx.data.MultiAlignedData(data_hparams['test'])
    iterator = tx.data.TrainTestDataIterator(train=train_data,
                                             val=val_data,
                                             test=test_data)
    data_batch = iterator.get_next()

    # (speaker's meta info)
    spk_src = tf.stack([data_batch['spk_{}'.format(i)]
                        for i in range(max_utterance_cnt)], 1)
    spk_tgt = data_batch['spk_tgt']
    def _add_source_speaker_token(x):
        return tf.concat([x, tf.reshape(spk_src, (-1, 1))], 1)
    def _add_target_speaker_token(x):
        return (x, ) + (tf.reshape(spk_tgt, (-1, 1)), )

    # HRED model
    embedder = tx.modules.WordEmbedder(
        init_value=train_data.embedding_init_value(0).word_vecs)
    encoder = tx.modules.HierarchicalRNNEncoder(hparams=encoder_hparams)

    decoder = tx.modules.BasicRNNDecoder(
        hparams=decoder_hparams, vocab_size=train_data.vocab(0).size)

    connector = tx.modules.connectors.MLPTransformConnector(
        decoder.cell.state_size)

    context_embed = embedder(data_batch['source_text_ids'])
    ecdr_states = encoder(
        context_embed,
        medium=['flatten', _add_source_speaker_token],
        sequence_length_minor=data_batch['source_length'],
        sequence_length_major=data_batch['source_utterance_cnt'])
    ecdr_states = ecdr_states[1]

    ecdr_states = _add_target_speaker_token(ecdr_states)
    dcdr_states = connector(ecdr_states)

    # (decoding for training)
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

    # Beam search
    target_bos_token_id = train_data.vocab(0).bos_token_id
    target_eos_token_id = train_data.vocab(0).eos_token_id
    start_tokens = \
        tf.ones_like(data_batch['target_length']) * target_bos_token_id

    beam_search_samples, beam_states, _ = tx.modules.beam_search_decode(
        decoder,
        initial_state=dcdr_states,
        start_tokens=start_tokens,
        end_token=target_eos_token_id,
        embedding=embedder,
        beam_width=config_model.beam_width,
        max_decoding_length=50)

    beam_lengths = beam_states.lengths
    beam_sample_text = train_data.vocab(0).map_ids_to_tokens(
        beam_search_samples.predicted_ids)

    # Running procedures

    def _train_epochs(sess, epoch, display=1000):
        iterator.switch_to_train_data(sess)

        for _ in range(3000):
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
                pples.append(ppl)

            except tf.errors.OutOfRangeError:
                avg_ppl = np.mean(pples)
                print('epoch {} perplexity={}'.format(epoch, avg_ppl))
                break

    def _test_epochs_bleu(sess, epoch):
        iterator.switch_to_test_data(sess)

        bleu_prec = [[] for i in range(1, 5)]
        bleu_recall = [[] for i in range(1, 5)]

        def _bleus(ref, sample):
            res = []
            for weight in [[1, 0, 0, 0],
                           [1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]]:
                res.append(sentence_bleu(
                    [ref],
                    sample,
                    smoothing_function=SmoothingFunction().method7,
                    weights=weight))
            return res

        while True:
            try:
                feed = {tx.global_mode(): tf.estimator.ModeKeys.EVAL}

                beam_samples, beam_length, references, refs_cnt = \
                    sess.run([beam_sample_text,
                              beam_lengths,
                              data_batch['refs_text'][:, :, 1:],
                              data_batch['refs_utterance_cnt']],
                             feed_dict=feed)

                beam_samples = np.transpose(beam_samples, (0, 2, 1))
                beam_samples = [
                    [sample[:l] for sample, l in zip(beam, lens)]
                    for beam, lens in zip(beam_samples.tolist(), beam_length)
                ]
                references = [
                    [ref[:ref.index(b'<EOS>')] for ref in refs[:cnt]]
                    for refs, cnt in zip(references.tolist(), refs_cnt)
                ]

                for beam, refs in zip(beam_samples, references):
                    bleu_scores = [
                        [_bleus(ref, sample) for ref in refs]
                        for sample in beam
                    ]
                    bleu_scores = np.transpose(np.array(bleu_scores), (2, 0, 1))

                    for i in range(1, 5):
                        bleu_i = bleu_scores[i]
                        bleu_i_precision = bleu_i.max(axis=1).mean()
                        bleu_i_recall = bleu_i.max(axis=0).mean()

                        bleu_prec[i-1].append(bleu_i_precision)
                        bleu_recall[i-1].append(bleu_i_recall)


            except tf.errors.OutOfRangeError:
                break

        bleu_prec = [np.mean(x) for x in bleu_prec]
        bleu_recall = [np.mean(x) for x in bleu_recall]

        print('epoch {}:'.format(epoch))
        for i in range(1, 5):
            print(' -- bleu-{} prec={}, recall={}'.format(
                i, bleu_prec[i-1], bleu_recall[i-1]))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        num_epochs = 10
        for epoch in range(num_epochs):
            _train_epochs(sess, epoch)
            _test_epochs_ppl(sess, epoch)

        _test_epochs_bleu(sess, num_epochs-1)

if __name__ == "__main__":
    main()
