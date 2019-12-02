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
Attentional Seq2seq with RAML algorithm.

Read a pre-processed file containing the augmented samples and
corresponding rewards for every target sentence.

RAML Algorithm is described in https://arxiv.org/pdf/1705.07136.pdf

"""

from io import open
import importlib
import tensorflow as tf
import texar.tf as tx
import numpy as np
import random
from rouge import Rouge

flags = tf.flags

flags.DEFINE_string("config_model", "configs.config_model", "The model config.")
flags.DEFINE_string("config_data", "configs.config_iwslt14",
                    "The dataset config.")

flags.DEFINE_string('raml_file', 'data/iwslt14/samples_iwslt14.txt',
                    'the samples and rewards described in RAML')
flags.DEFINE_integer('n_samples', 10,
                     'number of samples for every target sentence')
flags.DEFINE_float('tau', 0.4, 'the temperature in RAML algorithm')

flags.DEFINE_string('output_dir', '.', 'where to keep training logs')

FLAGS = flags.FLAGS

config_model = importlib.import_module(FLAGS.config_model)
config_data = importlib.import_module(FLAGS.config_data)

if not FLAGS.output_dir.endswith('/'):
    FLAGS.output_dir += '/'
log_dir = FLAGS.output_dir + 'training_log_raml' +\
          '_' + str(FLAGS.n_samples) + 'samples' +\
          '_tau' + str(FLAGS.tau) + '/'
tx.utils.maybe_create_dir(log_dir)


def read_raml_sample_file():
    raml_file = open(FLAGS.raml_file, encoding='utf-8')

    train_data = []
    sample_num = -1
    for line in raml_file.readlines():
        line = line[:-1]
        if line.startswith('***'):
            continue
        elif line.endswith('samples'):
            sample_num = eval(line.split()[0])
            assert sample_num == 1 or sample_num == FLAGS.n_samples
        elif line.startswith('source:'):
            train_data.append({'source': line[7:], 'targets': []})
        else:
            train_data[-1]['targets'].append(line.split('|||'))
            if sample_num == 1:
                for i in range(FLAGS.n_samples - 1):
                    train_data[-1]['targets'].append(line.split('|||'))
    return train_data


def raml_loss(batch, output, training_rewards):
    mle_loss = tx.losses.sequence_sparse_softmax_cross_entropy(
        labels=batch['target_text_ids'][:, 1:],
        logits=output.logits,
        sequence_length=batch['target_length'] - 1,
        average_across_batch=False)
    return tf.reduce_sum(mle_loss * training_rewards) /\
           tf.reduce_sum(training_rewards)


def build_model(batch, train_data, rewards):
    """
    Assembles the seq2seq model.
    Code in this function is basically the same of build_model() in
    baseline_seq2seq_attn_main.py except the normalization in loss_fn.
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
        raml_loss(batch, training_outputs, rewards),
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
    config_data.train['batch_size'] *= FLAGS.n_samples
    config_data.val['batch_size'] *= FLAGS.n_samples
    config_data.test['batch_size'] *= FLAGS.n_samples

    train_data = tx.data.PairedTextData(hparams=config_data.train)
    val_data = tx.data.PairedTextData(hparams=config_data.val)
    test_data = tx.data.PairedTextData(hparams=config_data.test)
    data_iterator = tx.data.TrainTestDataIterator(
        train=train_data, val=val_data, test=test_data)

    batch = data_iterator.get_next()
    rewards_ts = tf.placeholder(
        dtype=tf.float32, shape=[None, ], name='training_rewards')

    train_op, infer_outputs = build_model(batch, train_data, rewards_ts)

    raml_train_data = read_raml_sample_file()

    def _train_epoch(sess, epoch_no):
        data_iterator.switch_to_train_data(sess)
        training_log_file = \
            open(log_dir + 'training_log' + str(epoch_no) + '.txt', 'w',
                 encoding='utf-8')

        step = 0
        source_buffer, target_buffer = [], []
        random.shuffle(raml_train_data)
        for training_pair in raml_train_data:
            for target in training_pair['targets']:
                source_buffer.append(training_pair['source'])
                target_buffer.append(target)

            if len(target_buffer) != train_data.batch_size:
                continue

            source_ids = []
            source_length = []
            target_ids = []
            target_length = []
            scores = []

            trunc_len_src = train_data.hparams.source_dataset.max_seq_length
            trunc_len_tgt = train_data.hparams.target_dataset.max_seq_length

            for sentence in source_buffer:
                ids = [train_data.source_vocab.token_to_id_map_py[token]
                       for token in sentence.split()][:trunc_len_src]
                ids = ids + [train_data.source_vocab.eos_token_id]

                source_ids.append(ids)
                source_length.append(len(ids))

            for sentence, score_str in target_buffer:
                ids = [train_data.target_vocab.bos_token_id]
                ids = ids + [train_data.target_vocab.token_to_id_map_py[token]
                             for token in sentence.split()][:trunc_len_tgt]
                ids = ids + [train_data.target_vocab.eos_token_id]

                target_ids.append(ids)
                scores.append(eval(score_str))
                target_length.append(len(ids))

            rewards = []
            for i in range(0, train_data.batch_size, FLAGS.n_samples):
                tmp = np.array(scores[i:i + FLAGS.n_samples])
                tmp = np.exp(tmp / FLAGS.tau) / np.sum(np.exp(tmp / FLAGS.tau))
                for j in range(0, FLAGS.n_samples):
                    rewards.append(tmp[j])

            for value in source_ids:
                while len(value) < max(source_length):
                    value.append(0)
            for value in target_ids:
                while len(value) < max(target_length):
                    value.append(0)

            feed_dict = {
                batch['source_text_ids']: np.array(source_ids),
                batch['target_text_ids']: np.array(target_ids),
                batch['source_length']: np.array(source_length),
                batch['target_length']: np.array(target_length),
                rewards_ts: np.array(rewards)
            }
            source_buffer = []
            target_buffer = []

            loss = sess.run(train_op, feed_dict=feed_dict)
            print("step={}, loss={:.4f}".format(step, loss),
                  file=training_log_file)
            if step % config_data.observe_steps == 0:
                print("step={}, loss={:.4f}".format(step, loss))
            training_log_file.flush()
            step += 1

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
        scores_file = open(log_dir + 'scores.txt', 'w', encoding='utf-8')
        for i in range(config_data.num_epochs):
            _train_epoch(sess, i)

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
