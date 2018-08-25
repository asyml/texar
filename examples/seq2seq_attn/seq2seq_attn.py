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
"""Attentional Seq2seq.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

#pylint: disable=invalid-name, too-many-arguments, too-many-locals

import importlib
import tensorflow as tf
import texar as tx

flags = tf.flags

flags.DEFINE_string("config_model", "config_model", "The model config.")
flags.DEFINE_string("config_data", "config_iwslt14", "The dataset config.")

FLAGS = flags.FLAGS

config_model = importlib.import_module(FLAGS.config_model)
config_data = importlib.import_module(FLAGS.config_data)


def build_model(batch, train_data):
    """Assembles the seq2seq model.
    """
    source_embedder = tx.modules.WordEmbedder(
        vocab_size=train_data.source_vocab.size, hparams=config_model.embedder)

    encoder = tx.modules.BidirectionalRNNEncoder(
        hparams=config_model.encoder)

    enc_outputs, _ = encoder(source_embedder(batch['source_text_ids']))

    target_embedder = tx.modules.WordEmbedder(
        vocab_size=train_data.target_vocab.size, hparams=config_model.embedder)

    decoder = tx.modules.AttentionRNNDecoder(
        memory=tf.concat(enc_outputs, axis=2),
        memory_sequence_length=batch['source_length'],
        vocab_size=train_data.target_vocab.size,
        hparams=config_model.decoder)

    training_outputs, _, _ = decoder(
        decoding_strategy='train_greedy',
        inputs=target_embedder(batch['target_text_ids'][:, :-1]),
        sequence_length=batch['target_length'] - 1)

    train_op = tx.core.get_train_op(
        tx.losses.sequence_sparse_softmax_cross_entropy(
            labels=batch['target_text_ids'][:, 1:],
            logits=training_outputs.logits,
            sequence_length=batch['target_length'] - 1))

    start_tokens = tf.ones_like(batch['target_length']) * \
            train_data.target_vocab.bos_token_id
    beam_search_outputs, _, _ = \
        tx.modules.beam_search_decode(
            decoder_or_cell=decoder,
            embedding=target_embedder,
            start_tokens=start_tokens,
            end_token=train_data.target_vocab.eos_token_id,
            beam_width=config_model.beam_width,
            max_decoding_length=60)

    return train_op, beam_search_outputs


def main():
    """Entrypoint.
    """
    train_data = tx.data.PairedTextData(hparams=config_data.train)
    val_data = tx.data.PairedTextData(hparams=config_data.val)
    test_data = tx.data.PairedTextData(hparams=config_data.test)
    data_iterator = tx.data.TrainTestDataIterator(
        train=train_data, val=val_data, test=test_data)

    batch = data_iterator.get_next()

    train_op, infer_outputs = build_model(batch, train_data)

    def _train_epoch(sess):
        data_iterator.switch_to_train_data(sess)

        step = 0
        while True:
            try:
                loss = sess.run(train_op)
                if step % config_data.display == 0:
                    print("step={}, loss={:.4f}".format(step, loss))
                step += 1
            except tf.errors.OutOfRangeError:
                break

    def _eval_epoch(sess, mode):
        if mode == 'val':
            data_iterator.switch_to_val_data(sess)
        else:
            data_iterator.switch_to_test_data(sess)

        refs, hypos = [], []
        while True:
            try:
                fetches = [
                    batch['target_text'][:, 1:],
                    infer_outputs.predicted_ids[:, :, 0]
                ]
                feed_dict = {
                    tx.global_mode(): tf.estimator.ModeKeys.EVAL
                }
                target_texts_ori, output_ids = \
                    sess.run(fetches, feed_dict=feed_dict)

                target_texts = tx.utils.strip_special_tokens(target_texts_ori)
                output_texts = tx.utils.map_ids_to_strs(
                    ids=output_ids, vocab=val_data.target_vocab)

                for hypo, ref in zip(output_texts, target_texts):
                    hypos.append(hypo)
                    refs.append([ref])
            except tf.errors.OutOfRangeError:
                break

        return tx.evals.corpus_bleu_moses(list_of_references=refs,
                                          hypotheses=hypos)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        best_val_bleu = -1.
        for i in range(config_data.num_epochs):
            _train_epoch(sess)

            val_bleu = _eval_epoch(sess, 'val')
            best_val_bleu = max(best_val_bleu, val_bleu)
            print('val epoch={}, BLEU={:.4f}; best-ever={:.4f}'.format(
                i, val_bleu, best_val_bleu))

            test_bleu = _eval_epoch(sess, 'test')
            print('test epoch={}, BLEU={:.4f}'.format(i, test_bleu))

            print('=' * 50)


if __name__ == '__main__':
    main()

