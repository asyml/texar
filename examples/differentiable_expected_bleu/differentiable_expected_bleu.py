#!/usr/bin/env python3
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
import os
import tensorflow as tf
import texar as tx

from nltk.translate.bleu_score import corpus_bleu

flags = tf.flags

flags.DEFINE_string("config_model", "config_model_medium", "The model config.")
flags.DEFINE_string("config_data", "config_data_iwslt14_de-en",
                    "The dataset config.")
flags.DEFINE_string("config_train", "config_train", "The training config.")
flags.DEFINE_string("expr_name", "iwslt14_de-en", "The experiment name. "
                    "Used as the directory name of run.")
flags.DEFINE_boolean("reinitialize", True, "Whether to reinitialize the state "
                     "of the optimizers before training and after triggering.")

FLAGS = flags.FLAGS

config_model = importlib.import_module(FLAGS.config_model)
config_data = importlib.import_module(FLAGS.config_data)
config_train = importlib.import_module(FLAGS.config_train)
expr_name = FLAGS.expr_name
reinitialize = FLAGS.reinitialize
phases = config_train.phases

xe_names = ('xe_0', 'xe_1')
debleu_names = ('debleu_0', 'debleu_1')

def get_scope_by_name(tensor):
    return tensor.name[: tensor.name.rfind('/') + 1]


def build_model(batch, train_data):
    """Assembles the seq2seq model.
    """
    train_ops = {}

    source_embedder = tx.modules.WordEmbedder(
        vocab_size=train_data.source_vocab.size, hparams=config_model.embedder)

    encoder = tx.modules.BidirectionalRNNEncoder(
        hparams=config_model.encoder)

    enc_outputs, enc_final_state = encoder(
        source_embedder(batch['source_text_ids']))

    target_embedder = tx.modules.WordEmbedder(
        vocab_size=train_data.target_vocab.size, hparams=config_model.embedder)

    decoder = tx.modules.AttentionRNNDecoder(
        memory=tf.concat(enc_outputs, axis=2),
        memory_sequence_length=batch['source_length'],
        vocab_size=train_data.target_vocab.size,
        hparams=config_model.decoder)

    if config_model.connector is None:
        dec_initial_state = None

    else:
        enc_final_state = tf.contrib.framework.nest.map_structure(
            lambda *args: tf.concat(args, -1), *enc_final_state)

        if isinstance(decoder.cell, tf.nn.rnn_cell.LSTMCell):
            connector = tx.modules.MLPTransformConnector(
                decoder.state_size.h, hparams=config_model.connector)
            dec_initial_h = connector(enc_final_state.h)
            dec_initial_state = (dec_initial_h, enc_final_state.c)
        else:
            connector = tx.modules.MLPTransformConnector(
                decoder.state_size, hparams=config_model.connector)
            dec_initial_state = connector(enc_final_state)

    # cross-entropy + teacher-forcing pretraining
    tf_outputs, _, _ = decoder(
        decoding_strategy='train_greedy',
        initial_state=dec_initial_state,
        inputs=target_embedder(batch['target_text_ids'][:, :-1]),
        sequence_length=batch['target_length']-1)

    loss_xe = tx.losses.sequence_sparse_softmax_cross_entropy(
        labels=batch['target_text_ids'][:, 1:],
        logits=tf_outputs.logits,
        sequence_length=batch['target_length']-1)

    train_ops[xe_names[0]] = tx.core.get_train_op(
        loss_xe,
        hparams=config_train.train_xe_0)
    train_ops[xe_names[1]] = tx.core.get_train_op(
        loss_xe,
        hparams=config_train.train_xe_1)

    # teacher mask + DEBLEU fine-tuning
    tm_helper = tx.modules.TeacherMaskSoftmaxEmbeddingHelper(
        # must not remove last token, since it may be used as mask
        inputs=batch['target_text_ids'],
        sequence_length=batch['target_length']-1,
        embedding=target_embedder,
        n_unmask=1,
        n_mask=0,
        tau=config_train.tau)

    tm_outputs, _, _ = decoder(
        helper=tm_helper,
        initial_state=dec_initial_state)

    loss_debleu = tx.losses.debleu(
        labels=batch['target_text_ids'][:, 1:],
        probs=tm_outputs.sample_id,
        sequence_length=batch['target_length']-1)

    train_ops[debleu_names[0]] = tx.core.get_train_op(
        loss_debleu,
        hparams=config_train.train_debleu_0)
    train_ops[debleu_names[1]] = tx.core.get_train_op(
        loss_debleu,
        hparams=config_train.train_debleu_1)

    # inference: beam search decoding
    start_tokens = tf.ones_like(batch['target_length']) * \
            train_data.target_vocab.bos_token_id
    end_token = train_data.target_vocab.eos_token_id

    bs_outputs, _, _ = tx.modules.beam_search_decode(
        decoder_or_cell=decoder,
        embedding=target_embedder,
        start_tokens=start_tokens,
        end_token=end_token,
        initial_state=dec_initial_state,
        beam_width=config_train.infer_beam_width,
        max_decoding_length=config_train.infer_max_decoding_length)

    return train_ops, tm_helper, bs_outputs


def main():
    """Entrypoint.
    """
    train_0_data = tx.data.PairedTextData(hparams=config_data.train_0)
    train_1_data = tx.data.PairedTextData(hparams=config_data.train_1)
    val_data = tx.data.PairedTextData(hparams=config_data.val)
    test_data = tx.data.PairedTextData(hparams=config_data.test)
    data_iterator = tx.data.FeedableDataIterator(
        {'train_0': train_0_data, 'train_1': train_1_data,
         'val': val_data, 'test': test_data})
    data_batch = data_iterator.get_next()

    global_step = tf.train.create_global_step()

    train_ops, tm_helper, infer_outputs = build_model(data_batch, train_0_data)

    def get_train_op_scope(name):
        return get_scope_by_name(train_ops[name])

    train_op_initializers = {
        name: tf.variables_initializer(
            tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES,
                scope=get_train_op_scope(name)),
            name='train_{}_op_initializer'.format(name))
        for name in (xe_names + debleu_names)}
    tm_helper_initializer = tf.variables_initializer(
        [tm_helper.n_unmask, tm_helper.n_mask], name="tm_helper_initializer")

    summary_tm = [
        tf.summary.scalar('tm/n_unmask', tm_helper.n_unmask),
        tf.summary.scalar('tm/n_mask', tm_helper.n_mask)]
    summary_ops = {
        name: tf.summary.merge(
            tf.get_collection(
                tf.GraphKeys.SUMMARIES,
                scope=get_train_op_scope(name))
            + (summary_tm if name in debleu_names else []),
            name='summary_{}'.format(name))
        for name in (xe_names + debleu_names)}

    saver = tf.train.Saver(max_to_keep=None)

    def _restore_from(directory, restore_trigger):
        if os.path.exists(directory):
            ckpt_path = tf.train.latest_checkpoint(directory)
            print('restoring from {} ...'.format(ckpt_path))
            saver.restore(sess, ckpt_path)

            if restore_trigger:
                trigger_path = '{}.trigger'.format(ckpt_path)
                if os.path.exists(trigger_path):
                    with open(trigger_path, 'rb') as pickle_file:
                        trigger.restore_from_pickle(pickle_file)
                else:
                    print('cannot find previous trigger state.')

            print('done.')

        else:
            print('cannot find checkpoint directory {}'.format(directory))

    def _train_epoch(sess, summary_writer, mode, train_op, summary_op, trigger):
        print('in _train_epoch')

        data_iterator.restart_dataset(sess, mode)
        feed_dict = {
            tx.global_mode(): tf.estimator.ModeKeys.TRAIN,
            data_iterator.handle: data_iterator.get_handle(sess, mode)
        }

        while True:
            try:
                loss, summary, step = sess.run(
                    (train_op, summary_op, global_step), feed_dict)

                summary_writer.add_summary(summary, step)

                if step % config_train.steps_per_eval == 0:
                    global triggered
                    _eval_epoch(sess, summary_writer, 'val', trigger)
                    if triggered:
                        break

            except tf.errors.OutOfRangeError:
                break

        print('end _train_epoch')

    def _eval_epoch(sess, summary_writer, mode, trigger):
        print('in _eval_epoch with mode {}'.format(mode))

        data_iterator.restart_dataset(sess, mode)
        feed_dict = {
            tx.global_mode(): tf.estimator.ModeKeys.EVAL,
            data_iterator.handle: data_iterator.get_handle(sess, mode)
        }

        ref_hypo_pairs = []
        fetches = [
            data_batch['target_text'][:, 1:],
            infer_outputs.predicted_ids[:, :, 0]
        ]

        while True:
            try:
                target_texts_ori, output_ids = sess.run(fetches, feed_dict)
                target_texts = tx.utils.strip_special_tokens(
                    target_texts_ori.tolist(), is_token_list=True)
                output_texts = tx.utils.map_ids_to_strs(
                    ids=output_ids.tolist(), vocab=val_data.target_vocab,
                    join=False)

                ref_hypo_pairs.extend(
                    zip(map(lambda x: [x], target_texts), output_texts))

            except tf.errors.OutOfRangeError:
                break

        refs, hypos = zip(*ref_hypo_pairs)
        bleu = corpus_bleu(refs, hypos) * 100
        print('{} BLEU: {}'.format(mode, bleu))

        step = tf.train.global_step(sess, global_step)

        summary = tf.Summary()
        summary.value.add(tag='{}/BLEU'.format(mode), simple_value=bleu)
        summary_writer.add_summary(summary, step)
        summary_writer.flush()

        if mode == 'val':
            if trigger is not None:
                if (trigger.best_ever_score is not None and
                        bleu > trigger.best_ever_score):
                    print('update best val bleu: {}'.format(bleu))

                    saved_path = saver.save(sess, ckpt_best, global_step=step)
                    with open('{}.trigger'.format(saved_path), 'wb') as \
                            pickle_file:
                        trigger.save_to_pickle(pickle_file)
                    print('saved to {}'.format(saved_path))

                global triggered
                triggered, _ = trigger(step, bleu)
                if triggered:
                    print('triggered!')

        print('end _eval_epoch')
        return bleu

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        dir_model = os.path.join(expr_name, 'ckpt')
        dir_best = os.path.join(expr_name, 'ckpt-best')
        ckpt_model = os.path.join(dir_model, 'model.ckpt')
        ckpt_best = os.path.join(dir_best, 'model.ckpt')

        def action_before_phase(phase):
            global train_data_name, train_op_name, mask_pattern,\
                train_op, summary_op
            train_data_name, train_op_name, mask_pattern = phase
            train_op = train_ops[train_op_name]
            summary_op = summary_ops[train_op_name]
            if reinitialize:
                sess.run(train_op_initializers[train_op_name])
            if mask_pattern is not None:
                tm_helper.assign_mask_pattern(sess, *mask_pattern)

        action = (action_before_phase(phase) for phase in phases)
        next(action)
        trigger = tx.utils.BestEverConvergenceTrigger(
            action,
            config_train.threshold_steps,
            config_train.minimum_interval_steps,
            default=None)

        _restore_from(dir_model, restore_trigger=True)

        summary_writer = tf.summary.FileWriter(
            os.path.join(expr_name, 'log'), sess.graph, flush_secs=30)

        epoch = 0
        while epoch < config_train.max_epochs:
            print('epoch #{} {}:'.format(
                epoch, (train_data_name, train_op_name, mask_pattern)))

            val_bleu = _eval_epoch(sess, summary_writer, 'val', trigger)
            if triggered:
                _restore_from(dir_best, restore_trigger=False)

            test_bleu = _eval_epoch(sess, summary_writer, 'test', None)

            step = tf.train.global_step(sess, global_step)

            print('epoch: {}, step: {}, val BLEU: {}, test BLEU: {}'.format(
                epoch, step, val_bleu, test_bleu))

            _train_epoch(sess, summary_writer, train_data_name,
                         train_op, summary_op, trigger)
            if triggered:
                _restore_from(dir_best, restore_trigger=False)

            epoch += 1

            step = tf.train.global_step(sess, global_step)
            saved_path = saver.save(sess, ckpt_model, global_step=step)
            with open('{}.trigger'.format(saved_path), 'wb') as pickle_file:
                trigger.save_to_pickle(pickle_file)

            print('saved to {}'.format(saved_path))

        test_bleu = _eval_epoch(sess, summary_writer, 'test', None)
        print('epoch: {}, test BLEU: {}'.format(epoch, test_bleu))


if __name__ == '__main__':
    main()

