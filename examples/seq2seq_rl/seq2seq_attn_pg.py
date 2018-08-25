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
"""Attentional Seq2seq trained with policy gradient.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

#pylint: disable=invalid-name, too-many-arguments, too-many-locals

import importlib
import numpy as np
import tensorflow as tf
import texar as tx

flags = tf.flags

flags.DEFINE_string("config_model", "config_model", "The model config.")
flags.DEFINE_string("config_data", "config_iwslt14", "The dataset config.")

FLAGS = flags.FLAGS

config_model = importlib.import_module(FLAGS.config_model)
config_data = importlib.import_module(FLAGS.config_data)

# A caveats of using `texar.agents.SeqPGAgent`:
# The training data iterator should not run to raise `OutOfRangeError`,
# otherwise the iterator cannot be re-initialized and may raise
# `CancelledError`. This is probably because the iterator is used by
# `tf.Session.partial_run` in `SeqPGAgent`.
#
# A simple workaround is to set `'num_epochs'` of training data to a large
# number so that its iterator will never run into `OutOfRangeError`. Use
# `texar.data.FeedableDataIterator` to periodically switch to dev/test data
# for evaluation and switch back to the training data to resume from the
# breakpoint.

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

    start_tokens = tf.ones_like(batch['target_length']) * \
            train_data.target_vocab.bos_token_id

    outputs, _, sequence_length = decoder(
        decoding_strategy='infer_sample',
        start_tokens=start_tokens,
        end_token=train_data.target_vocab.eos_token_id,
        embedding=target_embedder,
        max_decoding_length=30)

    beam_search_outputs, _, _ = \
        tx.modules.beam_search_decode(
            decoder_or_cell=decoder,
            embedding=target_embedder,
            start_tokens=start_tokens,
            end_token=train_data.target_vocab.eos_token_id,
            beam_width=config_model.beam_width,
            max_decoding_length=60)

    return outputs, sequence_length, beam_search_outputs


def main():
    """Entrypoint.
    """
    train_data = tx.data.PairedTextData(hparams=config_data.train)
    val_data = tx.data.PairedTextData(hparams=config_data.val)
    test_data = tx.data.PairedTextData(hparams=config_data.test)
    iterator = tx.data.FeedableDataIterator(
        {'train': train_data, 'val': val_data, 'test': test_data})

    batch = iterator.get_next()

    outputs, sequence_length, infer_outputs = build_model(batch, train_data)

    agent = tx.agents.SeqPGAgent(
        samples=outputs.sample_id,
        logits=outputs.logits,
        sequence_length=sequence_length,
        hparams=config_model.agent)

    def _train_and_eval(sess, agent):
        iterator.restart_dataset(sess, 'train')

        best_val_bleu = -1.
        step = 0
        while True:
            try:
                # Samples
                extra_fetches = {
                    'truth': batch['target_text_ids'],
                }
                feed_dict = {
                    iterator.handle: iterator.get_handle(sess, 'train')
                }
                fetches = agent.get_samples(
                    extra_fetches=extra_fetches, feed_dict=feed_dict)

                sample_text = tx.utils.map_ids_to_strs(
                    fetches['samples'], train_data.target_vocab,
                    strip_eos=False, join=False)
                truth_text = tx.utils.map_ids_to_strs(
                    fetches['truth'], train_data.target_vocab,
                    strip_eos=False, join=False)

                # Computes rewards
                reward = []
                for ref, hyp in zip(truth_text, sample_text):
                    r = tx.evals.sentence_bleu([ref], hyp, smooth=True)
                    reward.append(r)

                # Updates
                loss = agent.observe(reward=reward)

                # Displays & evaluates
                step += 1
                if step == 1 or step % config_data.display == 0:
                    print("step={}, loss={:.4f}, reward={:.4f}".format(
                        step, loss, np.mean(reward)))

                if step % config_data.display_eval == 0:
                    val_bleu = _eval_epoch(sess, 'val')
                    best_val_bleu = max(best_val_bleu, val_bleu)
                    print('val step={}, BLEU={:.4f}; best-ever={:.4f}'.format(
                        step, val_bleu, best_val_bleu))

                    test_bleu = _eval_epoch(sess, 'test')
                    print('test step={}, BLEU={:.4f}'.format(step, test_bleu))
                    print('=' * 50)

            except tf.errors.OutOfRangeError:
                break

    def _eval_epoch(sess, mode):
        """`mode` is one of {'val', 'test'}
        """
        iterator.restart_dataset(sess, mode)

        refs, hypos = [], []
        while True:
            try:
                fetches = [
                    batch['target_text'][:, 1:],
                    infer_outputs.predicted_ids[:, :, 0]
                ]
                feed_dict = {
                    tx.global_mode(): tf.estimator.ModeKeys.PREDICT,
                    iterator.handle: iterator.get_handle(sess, mode)
                }
                target_texts, output_ids = \
                    sess.run(fetches, feed_dict=feed_dict)

                target_texts = tx.utils.strip_special_tokens(target_texts)
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

        agent.sess = sess

        _train_and_eval(sess, agent)

if __name__ == '__main__':
    main()
