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

from utils import data_utils, utils
from bleu_tool import bleu_wrapper
from utils.preprocess import bos_token_id, eos_token_id
# pylint: disable=invalid-name, too-many-locals

flags = tf.flags

flags.DEFINE_string("config_model", "config_model", "The model config.")
flags.DEFINE_string("config_data", "config_iwslt15", "The dataset config.")
flags.DEFINE_string("run_mode", "train_and_evaluate",
                    "Either train_and_evaluate or test.")
flags.DEFINE_string("model_dir", "./outputs",
                    "Directory to save the trained model and logs.")

FLAGS = flags.FLAGS

config_model = importlib.import_module(FLAGS.config_model)
config_data = importlib.import_module(FLAGS.config_data)

utils.set_random_seed(config_model.random_seed)

n_gpu = 2

def main():
    """Entrypoint.
    """
    # Load data
    train_data, dev_data, test_data = data_utils.load_data_numpy(
        config_data.input_dir, config_data.filename_prefix)
    with open(config_data.vocab_file, 'rb') as f:
        id2w = pickle.load(f)
    vocab_size = len(id2w)

    beam_width = config_model.beam_width

    # Create logging
    tx.utils.maybe_create_dir(FLAGS.model_dir)
    logging_file = os.path.join(FLAGS.model_dir, 'logging.txt')
    logger = utils.get_logger(logging_file)
    print('logging file is saved in: %s', logging_file)

    # Build model graph

    encoder_input_train = tf.placeholder(tf.int64, shape=(None, None))
    decoder_input_train = tf.placeholder(tf.int64, shape=(None, None))
    label_train = tf.placeholder(tf.int64, shape=(None, None))

    batch_size = tx.utils.get_batch_size(encoder_input_train)
    encoder_inputs = tf.split(encoder_input_train, n_gpu, 0)
    decoder_inputs = tf.split(decoder_input_train, n_gpu, 0)
    labels = tf.split(label_train, n_gpu, 0)

    encoder_input_test = tf.placeholder(tf.int64, shape=(None, None))
    decoder_input_test = tf.placeholder(tf.int64, shape=(None, None))
    label_test = tf.placeholder(tf.int64, shape=(None, None))

    embedder = tx.modules.WordEmbedder(
        vocab_size=vocab_size, hparams=config_model.emb)
    encoder = TransformerEncoder(hparams=config_model.encoder)
    # The decoder ties the input word embedding with the output logit layer.
    # As the decoder masks out <PAD>'s embedding, which in effect means
    # <PAD> has all-zero embedding, so here we explicitly set <PAD>'s embedding
    # to all-zero.
    tgt_embedding = tf.concat(
        [tf.zeros(shape=[1, embedder.dim]), embedder.embedding[1:, :]], axis=0)
    decoder = TransformerDecoder(embedding=tgt_embedding,
                                 hparams=config_model.decoder)
    global_step = tf.Variable(0, dtype=tf.int64, trainable=False)
    learning_rate = tf.placeholder(tf.float64, shape=(), name='lr')

    mle_losses = []
    for i, (encoder_input, decoder_input, label) in enumerate(zip(encoder_inputs, decoder_inputs, labels)):
        #with tf.device('gpu:{}'.format(i)), tf.variable_scope(tf.get_variable_scope(), reuse=i>0):
        #with tf.device('gpu:{}'.format(i)):
        with tf.variable_scope(tf.get_variable_scope()):
            #do_reuse = True if i>0 else None
            #print('i:{} reuse:{}'.format(i, do_reuse))
            encoder_input_length = tf.reduce_sum(
                1 - tf.to_int32(tf.equal(encoder_input, 0)), axis=1)
            decoder_input_length = tf.reduce_sum(
                1 - tf.to_int32(tf.equal(decoder_input, 0)), axis=1)
            is_target = tf.to_float(tf.not_equal(label, 0))

            encoder_output = encoder(inputs=embedder(encoder_input),
                                     sequence_length=encoder_input_length)

            # For training
            outputs = decoder(
                memory=encoder_output,
                memory_sequence_length=encoder_input_length,
                inputs=embedder(decoder_input),
                sequence_length=decoder_input_length,
                decoding_strategy='train_greedy',
                mode=tf.estimator.ModeKeys.TRAIN
            )

            mle_loss = transformer_utils.smoothing_cross_entropy(
                outputs.logits, label, vocab_size, config_model.loss_label_confidence)
            mle_loss = tf.reduce_sum(mle_loss * is_target) / tf.reduce_sum(is_target)
            mle_losses.append(mle_loss)
            print('trainable_variables:{}'.format(len(tf.trainable_variables())))
    mle_losses = tf.stack(mle_losses, axis=0)
    final_loss = tf.reduce_mean(mle_losses)
    #hparams = HParams(config_model.opt, default_optimization_hparams())['optimizer']
    #opt, _ = get_optimizer_fn(opt_hparams)
    train_op = tx.core.get_train_op(
        final_loss,
        learning_rate=learning_rate,
        global_step=global_step,
        hparams=config_model.opt)

    tf.summary.scalar('lr', learning_rate)
    tf.summary.scalar('mle_loss', mle_loss)
    summary_merged = tf.summary.merge_all()

    # For inference
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            # (text sequence length excluding padding)
        encoder_input_length = tf.reduce_sum(
            1 - tf.to_int32(tf.equal(encoder_input_test, 0)), axis=1)
        decoder_input_length = tf.reduce_sum(
            1 - tf.to_int32(tf.equal(decoder_input_test, 0)), axis=1)
        is_target = tf.to_float(tf.not_equal(label_test, 0))

        encoder_output = encoder(inputs=embedder(encoder_input_test),
                                 sequence_length=encoder_input_length)

        # The decoder ties the input word embedding with the output logit layer.
        # As the decoder masks out <PAD>'s embedding, which in effect means
        # <PAD> has all-zero embedding, so here we explicitly set <PAD>'s embedding
        # to all-zero.
        # For training
        start_tokens = tf.fill([tx.utils.get_batch_size(encoder_input_test)],
                               bos_token_id)
        predictions = decoder(
            memory=encoder_output,
            memory_sequence_length=encoder_input_length,
            decoding_strategy='infer_greedy',
            beam_width=beam_width,
            alpha=config_model.alpha,
            start_tokens=start_tokens,
            end_token=eos_token_id,
            max_decoding_length=config_data.max_decoding_length,
            mode=tf.estimator.ModeKeys.PREDICT
        )
        if beam_width <= 1:
            inferred_ids = predictions[0].sample_id
        else:
            # Uses the best sample by beam search
            inferred_ids = predictions['sample_id'][:, :, 0]

    #with tf.device('cpu:0'):
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
            fname = os.path.join(FLAGS.model_dir, 'tmp.eval')
            hypotheses = tx.utils.str_join(hypotheses)
            references = tx.utils.str_join(references)
            hyp_fn, ref_fn = tx.utils.write_paired_text(
                hypotheses, references, fname, mode='s')
            eval_bleu = bleu_wrapper(ref_fn, hyp_fn, case_sensitive=True)
            eval_bleu = 100. * eval_bleu
            logger.info('epoch: %d, eval_bleu %.4f', epoch, eval_bleu)
            print('epoch: %d, eval_bleu %.4f' % (epoch, eval_bleu))

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
            fname = os.path.join(FLAGS.model_dir, 'test.output')
            hwords, rwords = [], []
            for hyp, ref in zip(hypotheses, references):
                hwords.append([id2w[y] for y in hyp])
                rwords.append([id2w[y] for y in ref])
            hwords = tx.utils.str_join(hwords)
            rwords = tx.utils.str_join(rwords)
            hyp_fn, ref_fn = tx.utils.write_paired_text(
                hwords, rwords, fname, mode='s')
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
            print('train_batch:{} {}'.format(in_arrays[0].shape, in_arrays[1].shape))
            feed_dict = {
                encoder_input_train: in_arrays[0],
                decoder_input_train: in_arrays[1],
                label_train: in_arrays[2],
                learning_rate: utils.get_lr(step, config_model.lr)
            }
            fetches = {
                'step': global_step,
                'train_op': train_op,
                'smry': summary_merged,
                'loss': final_loss,
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
    sess_config = tf.ConfigProto(allow_soft_placement=True,
                                 log_device_placement=True)
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
