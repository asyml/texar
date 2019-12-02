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
"""
Attentional Seq2seq using Scheduled sampling algorithm.

This code is basically the same as baseline_seq2seq_attn_main.py,
except using ScheduledEmbeddingTrainingHelper.

Scheduled Sampling Algorithm is described in https://arxiv.org/abs/1506.03099
"""

# pylint: disable=invalid-name, too-many-arguments, too-many-locals

from io import open
import math
import importlib
import tensorflow as tf
import texar.tf as tx
from rouge import Rouge

flags = tf.flags

flags.DEFINE_string("config_model", "configs.config_model", "The model config.")
flags.DEFINE_string("config_data", "configs.config_iwslt14",
                    "The dataset config.")

flags.DEFINE_float('decay_factor', 500.,
                   'The hyperparameter controling the speed of increasing '
                   'the probability of sampling from model')

flags.DEFINE_string('output_dir', '.', 'where to keep training logs')

FLAGS = flags.FLAGS

config_model = importlib.import_module(FLAGS.config_model)
config_data = importlib.import_module(FLAGS.config_data)

if not FLAGS.output_dir.endswith('/'):
    FLAGS.output_dir += '/'
log_dir = FLAGS.output_dir + 'training_log_scheduled_sampling' +\
          '_decayf' + str(FLAGS.decay_factor) + '/'
tx.utils.maybe_create_dir(log_dir)


def inverse_sigmoid(i):
    return FLAGS.decay_factor / (
            FLAGS.decay_factor + math.exp(i / FLAGS.decay_factor))


def build_model(batch, train_data, self_sampling_proba):
    """
    Assembles the seq2seq model.
    It is the same as build_model() in baseline_seq2seq_attn.py except
    using ScheduledEmbeddingTrainingHelper.
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

    helper = tx.modules.get_helper(
        helper_type='ScheduledEmbeddingTrainingHelper',
        inputs=target_embedder(batch['target_text_ids'][:, :-1]),
        sequence_length=batch['target_length'] - 1,
        embedding=target_embedder,
        sampling_probability=self_sampling_proba)

    training_outputs, _, _ = decoder(
        helper=helper, initial_state=decoder.zero_state(
            batch_size=tf.shape(batch['target_length'])[0], dtype=tf.float32))

    train_op = tx.core.get_train_op(
        tx.losses.sequence_sparse_softmax_cross_entropy(
            labels=batch['target_text_ids'][:, 1:],
            logits=training_outputs.logits,
            sequence_length=batch['target_length'] - 1),
        hparams=config_model.opt)

    start_tokens = tf.ones_like(batch['target_length']) *\
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


def print_stdout_and_file(content, file):
    print(content)
    print(content, file=file)


def main():
    """Entrypoint.
    """
    train_data = tx.data.PairedTextData(hparams=config_data.train)
    val_data = tx.data.PairedTextData(hparams=config_data.val)
    test_data = tx.data.PairedTextData(hparams=config_data.test)
    data_iterator = tx.data.TrainTestDataIterator(
        train=train_data, val=val_data, test=test_data)

    batch = data_iterator.get_next()

    self_sampling_proba = tf.placeholder(shape=[], dtype=tf.float32)
    train_op, infer_outputs = \
        build_model(batch, train_data, self_sampling_proba)

    def _train_epoch(sess, epoch_no, total_step_counter):
        data_iterator.switch_to_train_data(sess)
        training_log_file = \
            open(log_dir + 'training_log' + str(epoch_no) + '.txt', 'w',
                 encoding='utf-8')

        step = 0
        while True:
            try:
                sampling_proba_ = 1. - inverse_sigmoid(total_step_counter)
                loss = sess.run(train_op, feed_dict={
                    self_sampling_proba: sampling_proba_})
                print("step={}, loss={:.4f}, self_proba={}".format(
                    step, loss, sampling_proba_), file=training_log_file)
                if step % config_data.observe_steps == 0:
                    print("step={}, loss={:.4f}, self_proba={}".format(
                        step, loss, sampling_proba_))
                training_log_file.flush()
                step += 1
                total_step_counter += 1
            except tf.errors.OutOfRangeError:
                break

    # code below this line is exactly the same as baseline_seq2seq_attn_main.py

    def _eval_epoch(sess, mode, epoch_no):
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

                target_texts = tx.utils.strip_special_tokens(
                    target_texts_ori.tolist(), is_token_list=True)
                target_texts = tx.utils.str_join(target_texts)
                output_texts = tx.utils.map_ids_to_strs(
                    ids=output_ids, vocab=val_data.target_vocab)

                tx.utils.write_paired_text(
                    target_texts, output_texts,
                    log_dir + mode + '_results' + str(epoch_no) + '.txt',
                    append=True, mode='h', sep=' ||| ')

                for hypo, ref in zip(output_texts, target_texts):
                    if config_data.eval_metric == 'bleu':
                        hypos.append(hypo)
                        refs.append([ref])
                    elif config_data.eval_metric == 'rouge':
                        hypos.append(tx.utils.compat_as_text(hypo))
                        refs.append(tx.utils.compat_as_text(ref))
            except tf.errors.OutOfRangeError:
                break

        if config_data.eval_metric == 'bleu':
            return tx.evals.corpus_bleu_moses(
                list_of_references=refs, hypotheses=hypos)
        elif config_data.eval_metric == 'rouge':
            rouge = Rouge()
            return rouge.get_scores(hyps=hypos, refs=refs, avg=True)

    def _calc_reward(score):
        """
        Return the bleu score or the sum of (Rouge-1, Rouge-2, Rouge-L).
        """
        if config_data.eval_metric == 'bleu':
            return score
        elif config_data.eval_metric == 'rouge':
            return sum([value['f'] for key, value in score.items()])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        best_val_score = -1.
        total_step_counter = 1
        scores_file = open(log_dir + 'scores.txt', 'w', encoding='utf-8')
        for i in range(config_data.num_epochs):
            _train_epoch(sess, i, total_step_counter)

            val_score = _eval_epoch(sess, 'val', i)
            test_score = _eval_epoch(sess, 'test', i)

            best_val_score = max(best_val_score, _calc_reward(val_score))

            if config_data.eval_metric == 'bleu':
                print_stdout_and_file(
                    'val epoch={}, BLEU={:.4f}; best-ever={:.4f}'.format(
                        i, val_score, best_val_score), file=scores_file)

                print_stdout_and_file(
                    'test epoch={}, BLEU={:.4f}'.format(i, test_score),
                    file=scores_file)
                print_stdout_and_file('=' * 50, file=scores_file)

            elif config_data.eval_metric == 'rouge':
                print_stdout_and_file(
                    'valid epoch {}:'.format(i), file=scores_file)
                for key, value in val_score.items():
                    print_stdout_and_file(
                        '{}: {}'.format(key, value), file=scores_file)
                print_stdout_and_file('fsum: {}; best_val_fsum: {}'.format(
                    _calc_reward(val_score), best_val_score), file=scores_file)

                print_stdout_and_file(
                    'test epoch {}:'.format(i), file=scores_file)
                for key, value in test_score.items():
                    print_stdout_and_file(
                        '{}: {}'.format(key, value), file=scores_file)
                print_stdout_and_file('=' * 110, file=scores_file)

            scores_file.flush()


if __name__ == '__main__':
    main()
