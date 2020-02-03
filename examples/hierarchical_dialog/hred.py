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

# pylint: disable=invalid-name, too-many-locals

import importlib
import numpy as np
import tensorflow as tf
import texar.tf as tx

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

flags = tf.flags

flags.DEFINE_string('config_data', 'config_data', 'The data config')
flags.DEFINE_string('config_model', 'config_model_biminor', 'The model config')

FLAGS = flags.FLAGS

config_data = importlib.import_module(FLAGS.config_data)
config_model = importlib.import_module(FLAGS.config_model)

encoder_hparams = config_model.encoder_hparams
decoder_hparams = config_model.decoder_hparams
opt_hparams = config_model.opt_hparams


def main():
    """Entrypoint.
    """
    # Data
    train_data = tx.data.MultiAlignedData(config_data.data_hparams['train'])
    val_data = tx.data.MultiAlignedData(config_data.data_hparams['val'])
    test_data = tx.data.MultiAlignedData(config_data.data_hparams['test'])
    iterator = tx.data.TrainTestDataIterator(train=train_data,
                                             val=val_data,
                                             test=test_data)
    data_batch = iterator.get_next()

    # (speaker's meta info)
    spk_src = tf.stack([data_batch['spk_{}'.format(i)]
                        for i in range(config_data.max_utterance_cnt)], 1)
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

    # Sentence level lld, for training
    mle_loss = tx.losses.sequence_sparse_softmax_cross_entropy(
        labels=data_batch['target_text_ids'][:, 1:],
        logits=outputs.logits,
        sequence_length=lengths)
    # Token level lld, for perplexity evaluation
    avg_mle_loss = tx.losses.sequence_sparse_softmax_cross_entropy(
        labels=data_batch['target_text_ids'][:, 1:],
        logits=outputs.logits,
        sequence_length=lengths,
        sum_over_timesteps=False,
        average_across_timesteps=True)
    perplexity = tf.exp(avg_mle_loss)

    global_step = tf.Variable(0, name='global_step', trainable=True)
    train_op = tx.core.get_train_op(
        mle_loss, global_step=global_step, hparams=opt_hparams)

    # Decoding

    target_bos_token_id = train_data.vocab(0).bos_token_id
    target_eos_token_id = train_data.vocab(0).eos_token_id
    start_tokens = \
        tf.ones_like(data_batch['target_length']) * target_bos_token_id

    # Random sample decoding
    decoding_strategy = 'infer_' + 'sample'
    infer_samples, lengths = [], []
    for _ in range(config_model.num_samples):
        infer_outputs_i, _, lengths_i = decoder(
            decoding_strategy=decoding_strategy,
            initial_state=dcdr_states,
            start_tokens=start_tokens,
            end_token=target_eos_token_id,
            embedding=embedder,
            max_decoding_length=50)
        infer_samples.append(
            tf.expand_dims(infer_outputs_i.sample_id, axis=2))
        lengths.append(tf.expand_dims(lengths_i, axis=1))

    infer_samples = tx.utils.pad_and_concat(
        infer_samples, axis=2, pad_axis=1)
    rand_sample_text = train_data.vocab(0).map_ids_to_tokens(infer_samples)
    rand_lengths = tf.concat(lengths, axis=1)

    # Beam search decoding
    beam_search_samples, beam_states, _ = tx.modules.beam_search_decode(
        decoder,
        initial_state=dcdr_states,
        start_tokens=start_tokens,
        end_token=target_eos_token_id,
        embedding=embedder,
        beam_width=config_model.beam_width,
        max_decoding_length=50)

    beam_sample_text = train_data.vocab(0).map_ids_to_tokens(
        beam_search_samples.predicted_ids)
    beam_lengths = beam_states.lengths

    # Running procedures

    def _train_epoch(sess, epoch, display=1000):
        iterator.switch_to_train_data(sess)

        while True:
            try:
                feed = {tx.global_mode(): tf.estimator.ModeKeys.TRAIN}
                step, loss, _ = sess.run(
                    [global_step, mle_loss, train_op], feed_dict=feed)

                if step % display == 0:
                    print('step {} at epoch {}: loss={}'.format(
                        step, epoch, loss))

            except tf.errors.OutOfRangeError:
                break

        print('epoch {} train: loss={}'.format(epoch, loss))

    def _test_epoch_ppl(sess, epoch):
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

    def _test_epoch_bleu(sess, epoch, sample_text, sample_lengths):
        iterator.switch_to_test_data(sess)

        bleu_prec = [[] for i in range(1, 5)]
        bleu_recall = [[] for i in range(1, 5)]

        def _bleus(ref, sample):
            res = []
            for weight in [[1, 0, 0, 0],
                           [1, 0, 0, 0],
                           [1 / 2., 1 / 2., 0, 0],
                           [1 / 3., 1 / 3., 1 / 3., 0],
                           [1 / 4., 1 / 4., 1 / 4., 1 / 4.]]:
                res.append(sentence_bleu(
                    [ref],
                    sample,
                    smoothing_function=SmoothingFunction().method7,
                    weights=weight))
            return res

        while True:
            try:
                feed = {tx.global_mode(): tf.estimator.ModeKeys.EVAL}

                samples_, sample_lengths_, references, refs_cnt = \
                    sess.run([sample_text,
                              sample_lengths,
                              data_batch['refs_text'][:, :, 1:],
                              data_batch['refs_utterance_cnt']],
                             feed_dict=feed)

                samples_ = np.transpose(samples_, (0, 2, 1))
                samples_ = [
                    [sample[:l] for sample, l in zip(beam, lens)]
                    for beam, lens in zip(samples_.tolist(), sample_lengths_)
                ]
                references = [
                    [ref[:ref.index(b'<EOS>')] for ref in refs[:cnt]]
                    for refs, cnt in zip(references.tolist(), refs_cnt)
                ]

                for beam, refs in zip(samples_, references):
                    bleu_scores = [
                        [_bleus(ref, sample) for ref in refs]
                        for sample in beam
                    ]
                    bleu_scores = np.transpose(np.array(bleu_scores), (2, 0, 1))

                    for i in range(1, 5):
                        bleu_i = bleu_scores[i]
                        bleu_i_precision = bleu_i.max(axis=1).mean()
                        bleu_i_recall = bleu_i.max(axis=0).mean()

                        bleu_prec[i - 1].append(bleu_i_precision)
                        bleu_recall[i - 1].append(bleu_i_recall)

            except tf.errors.OutOfRangeError:
                break

        bleu_prec = [np.mean(x) for x in bleu_prec]
        bleu_recall = [np.mean(x) for x in bleu_recall]

        print('epoch {}:'.format(epoch))
        for i in range(1, 5):
            print(' -- bleu-{} prec={}, recall={}'.format(
                i, bleu_prec[i - 1], bleu_recall[i - 1]))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        num_epochs = 10
        for epoch in range(1, num_epochs + 1):
            _train_epoch(sess, epoch)
            _test_epoch_ppl(sess, epoch)

            if epoch % 5 == 0:
                print('random sample: ')
                _test_epoch_bleu(sess, epoch, rand_sample_text, rand_lengths)
                print('beam-search: ')
                _test_epoch_bleu(sess, epoch, beam_sample_text, beam_lengths)

        if num_epochs % 5 != 0:
            print('random sample: ')
            _test_epoch_bleu(sess, num_epochs, rand_sample_text, rand_lengths)
            print('beam-search: ')
            _test_epoch_bleu(sess, num_epochs, beam_sample_text, beam_lengths)


if __name__ == "__main__":
    main()
