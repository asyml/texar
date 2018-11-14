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
"""Transformer model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pickle
import random
import os
import importlib
from torchtext import data
import numpy as np
import tensorflow as tf
import texar as tx
from texar.modules import TransformerEncoder, TransformerDecoder
from texar.utils import transformer_utils
from texar.utils.mode import is_train_mode

from utils import data_utils, utils
from bleu_tool import bleu_wrapper
from utils.preprocess import bos_token_id, eos_token_id
# pylint: disable=invalid-name, too-many-locals

flags = tf.flags

flags.DEFINE_string("config_model", "config_model_bert", "The model config.")
flags.DEFINE_string("config_data", "config_iwslt15", "The dataset config.")
flags.DEFINE_string("run_mode", "train_and_evaluate",
                    "Either train_and_evaluate or test.")
flags.DEFINE_string("model_dir", "./outputs",
                    "Directory to save the trained model and logs.")

FLAGS = flags.FLAGS

config_model = importlib.import_module(FLAGS.config_model)
utils.set_random_seed(config_model.random_seed)

config_data = importlib.import_module(FLAGS.config_data)

n_gpu = config_data.n_gpu

def main():
    """Entrypoint.
    """
    # Load data
    train_data, dev_data, test_data = data_utils.load_data_numpy(
        config_data.input_dir, config_data.filename_prefix)
    with open(config_data.vocab_file, 'rb') as f:
        id2w = pickle.load(f)

    beam_width = config_model.beam_width

    # Create logging
    tx.utils.maybe_create_dir(FLAGS.model_dir)
    logging_file = os.path.join(FLAGS.model_dir, 'logging.txt')
    logger = utils.get_logger(logging_file)
    print('logging file is saved in: %s', logging_file)

    # Build model graph
    input_ids = tf.placeholder(tf.int64, shape=(None, None))
    input_length = tf.reduce_sum(
        1 - tf.to_int32(tf.equal(input_ids, 0)), axis=1)
    embedder = tx.modules.WordEmbedder(
        vocab_size=28996, hparams=config_model.emb)
    token_type_embedder = tx.modules.WordEmbedder(
        vocab_size=2, hparams=config_model.token_embed)

    word_embeds = embedder(input_ids)
    token_type_ids = tf.zeros_like(input_ids)
    token_type_embeds = token_type_embedder(token_type_ids)

    input_embeds = word_embeds + token_type_embeds

    encoder = TransformerEncoder(hparams=config_model.encoder)

    pooled_output = encoder(input_embeds, input_length)
    num_labels = 2
    labels = tf.placeholder(tf.int64, shape=(None, None))
    output_weights = tf.get_variable(
        "output_weights", [num_labels, 768],
        initializer=tf.truncated_normal_initializer(stddev=0.02)
    )
    output_bias = tf.get_variable(
        "output_bias", [num_labels],
        initializer=tf.zeros_initializer()
    )

    output_layer = tf.layers.dropout(
        pooled_output,
        rate=0.1,
        training=is_train_mode(None)
    )
    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
    per_example_loss = - tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)

    _vars =  tf.trainable_variables()
    for var in _vars:
        print('var.name:{} var.shape:{}'.format(var.name, var.shape))
    exit()

    mle_loss = tf.reduce_sum(mle_losses * is_targets) / tf.reduce_sum(is_targets)
    #hparams = HParams(config_model.opt, default_optimization_hparams())['optimizer']
    #opt, _ = get_optimizer_fn(opt_hparams)
    train_op = tx.core.get_train_op(
        mle_loss,
        learning_rate=learning_rate,
        global_step=global_step,
        hparams=config_model.opt)

    tf.summary.scalar('lr', learning_rate)
    tf.summary.scalar('mle_loss', mle_loss)
    summary_merged = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=5)
    best_results = {'score': 0, 'epoch': -1}

    def _eval_epoch(sess, epoch, mode):
        if mode == 'eval':
            eval_data = dev_data
        elif mode == 'test':
            eval_data = test_data
        else:
            raise ValueError('`mode` should be either "eval" or "test".')

        references, hypotheses = [], []
        bsize = config_data.test_batch_size
        for i in range(0, len(eval_data), bsize):
            sources, targets = zip(*eval_data[i:i+bsize])
            x_block = data_utils.source_pad_concat_convert(sources)
            print('encoder_input_test:{}'.format(x_block.shape))
            logger.info('encoder_input_test:{}'.format(x_block.shape))
            feed_dict = {
                encoder_input_test: x_block,
                tx.global_mode(): tf.estimator.ModeKeys.PREDICT,
            }
            fetches = {
                'inferred_ids': inferred_ids,
            }
            fetches_ = sess.run(fetches, feed_dict=feed_dict)

            hypotheses.extend(h.tolist() for h in fetches_['inferred_ids'])
            references.extend(r.tolist() for r in targets)
            hypotheses = utils.list_strip_eos(hypotheses, eos_token_id)
            references = utils.list_strip_eos(references, eos_token_id)

        if mode == 'eval':
            # Writes results to files to evaluate BLEU
            # For 'eval' mode, the BLEU is based on token ids (rather than
            # text tokens) and serves only as a surrogate metric to monitor
            # the training process
            fname = os.path.join(FLAGS.model_dir, 'tmp.eval.digit')
            _hypotheses = tx.utils.str_join(hypotheses)
            _references = tx.utils.str_join(references)
            hyp_fn, ref_fn = tx.utils.write_paired_text(
                _hypotheses, _references, fname, mode='s')
            eval_bleu = bleu_wrapper(ref_fn, hyp_fn, case_sensitive=True)
            eval_bleu = 100. * eval_bleu
            logger.info('epoch: %d, eval_bleu_digit %.4f', epoch, eval_bleu)
            print('epoch: %d, eval_bleu_digit %.4f' % (epoch, eval_bleu))

            fname = os.path.join(FLAGS.model_dir, 'tmp.eval.txt')
            for hyp, ref in zip(hypotheses, references):
                hwords.append([id2w[y] for y in hyp])
                rwords.append([id2w[y] for y in ref])
            hwords = tx.utils.str_join(hwords)
            rwords = tx.utils.str_join(rwords)
            hyp_fn, ref_fn = tx.utils.write_paired_text(
                hwords, rwords, fname, mode='s')
            eval_bleu = bleu_wrapper(ref_fn, hyp_fn, case_sensitive=True)
            eval_bleu = 100. * eval_bleu
            logger.info('epoch: %d, eval_bleu txt %.4f', epoch, eval_bleu)
            print('epoch: %d, eval_bleu txt %.4f' % (epoch, eval_bleu))

            if eval_bleu > best_results['score']:
                logger.info('epoch: %d, best bleu: %.4f', epoch, eval_bleu)
                best_results['score'] = eval_bleu
                best_results['epoch'] = epoch
                model_path = os.path.join(FLAGS.model_dir, 'best-model.ckpt')
                logger.info('saving model to %s', model_path)
                print('saving model to %s' % model_path)
                saver.save(sess, model_path)

        elif mode == 'test':
            # For 'test' mode, together with the cmds in README.md, BLEU
            # is evaluated based on text tokens, which is the standard metric.
            hwords, rwords = [], []
            fname = os.path.join(FLAGS.model_dir, 'test.output.digit')
            _hypotheses = tx.utils.str_join(hypotheses)
            _references = tx.utils.str_join(references)
            hyp_fn, ref_fn = tx.utils.write_paired_text(
                _hypotheses, _references, fname, mode='s')
            eval_bleu = bleu_wrapper(ref_fn, hyp_fn, case_sensitive=True)
            eval_bleu = 100. * eval_bleu
            logger.info('test bleu_digit %.4f', epoch, eval_bleu)
            print('test bleu_digit %.4f' % (epoch, eval_bleu))

            fname = os.path.join(FLAGS.model_dir, 'test.output')
            for hyp, ref in zip(hypotheses, references):
                hwords.append([id2w[y] for y in hyp])
                rwords.append([id2w[y] for y in ref])
            hwords = tx.utils.str_join(hwords)
            rwords = tx.utils.str_join(rwords)
            hyp_fn, ref_fn = tx.utils.write_paired_text(
                hwords, rwords, fname, mode='s')
            eval_bleu = bleu_wrapper(ref_fn, hyp_fn, case_sensitive=True)
            eval_bleu = 100. * eval_bleu
            logger.info('test_bleu for raw text %.4f', epoch, eval_bleu)
            print('test bleu for raw text %.4f' % (epoch, eval_bleu))
            logger.info('Test output writtn to file: %s', hyp_fn)
            print('Test output writtn to file: %s' % hyp_fn)

    def _train_epoch(sess, epoch, step, smry_writer):
        random.shuffle(train_data)
        train_iter = data.iterator.pool(
            train_data,
            config_data.batch_size,
            key=lambda x: (len(x[0]), len(x[1])),
            batch_size_fn=utils.batch_size_fn,
            random_shuffler=data.iterator.RandomShuffler())

        for _, train_batch in enumerate(train_iter):
            in_arrays = data_utils.seq2seq_pad_concat_convert(train_batch, n_gpu=n_gpu)
            feed_dict = {
                encoder_input_train: in_arrays[0],
                decoder_input_train: in_arrays[1],
                label_train: in_arrays[2],
                learning_rate: utils.get_lr(step, config_model.lr)
            }
            print('encoder_input_train:{} decoder_input_train:{} label_train:{}'.format(in_arrays[0].shape, in_arrays[1].shape, in_arrays[2].shape))
            logger.info('encoder_input_train:{} decoder_input_train:{} label_train:{}'.format(in_arrays[0].shape, in_arrays[1].shape, in_arrays[2].shape))
            fetches = {
                'step': global_step,
                'train_op': train_op,
                'smry': summary_merged,
                'loss': mle_loss,
            }

            fetches_ = sess.run(fetches, feed_dict=feed_dict)

            step, loss = fetches_['step'], fetches_['loss']
            if step and step % config_data.display_steps == 0:
                logger.info('step: %d, loss: %.4f', step, loss)
                print('step: %d, loss: %.4f' % (step, loss))
                smry_writer.add_summary(fetches_['smry'], global_step=step)

            if step and step % config_data.eval_steps == 0:
                _eval_epoch(sess, epoch, mode='eval')
        return step
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    #log_device_placement=True)
    # Run the graph
    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        smry_writer = tf.summary.FileWriter(FLAGS.model_dir, graph=sess.graph)

        if FLAGS.run_mode == 'train_and_evaluate':
            logger.info('Begin running with train_and_evaluate mode')
            step = 0
            for epoch in range(config_data.max_train_epoch):
                step = _train_epoch(sess, epoch, step, smry_writer)

        elif FLAGS.run_mode == 'test':
            logger.info('Begin running with test mode')
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.model_dir))
            _eval_epoch(sess, 0, mode='test')


if __name__ == '__main__':
    main()
