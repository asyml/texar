# Copyright 2019 The Texar Authors. All Rights Reserved.
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

import pickle
import random
import os
import importlib
import tensorflow as tf
from torchtext import data
import texar.tf as tx
from texar.tf.modules import TransformerEncoder, TransformerDecoder
from texar.tf.utils import transformer_utils

from utils import data_utils, utils
from utils.preprocess import bos_token_id, eos_token_id
from bleu_tool import bleu_wrapper

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
    encoder_input = tf.placeholder(tf.int64, shape=(None, None))
    decoder_input = tf.placeholder(tf.int64, shape=(None, None))
    batch_size = tf.shape(encoder_input)[0]
    # (text sequence length excluding padding)
    encoder_input_length = tf.reduce_sum(
        1 - tf.cast(tf.equal(encoder_input, 0), tf.int32), axis=1)

    labels = tf.placeholder(tf.int64, shape=(None, None))
    is_target = tf.cast(tf.not_equal(labels, 0), tf.float32)

    global_step = tf.Variable(0, dtype=tf.int64, trainable=False)
    learning_rate = tf.placeholder(tf.float64, shape=(), name='lr')

    # Source word embedding
    src_word_embedder = tx.modules.WordEmbedder(
        vocab_size=vocab_size, hparams=config_model.emb)
    src_word_embeds = src_word_embedder(encoder_input)
    src_word_embeds = src_word_embeds * config_model.hidden_dim ** 0.5

    # Position embedding (shared b/w source and target)
    pos_embedder = tx.modules.SinusoidsPositionEmbedder(
        position_size=config_data.max_decoding_length,
        hparams=config_model.position_embedder_hparams)
    src_seq_len = tf.ones([batch_size], tf.int32) * tf.shape(encoder_input)[1]
    src_pos_embeds = pos_embedder(sequence_length=src_seq_len)

    src_input_embedding = src_word_embeds + src_pos_embeds

    encoder = TransformerEncoder(hparams=config_model.encoder)
    encoder_output = encoder(inputs=src_input_embedding,
                             sequence_length=encoder_input_length)

    # The decoder ties the input word embedding with the output logit layer.
    # As the decoder masks out <PAD>'s embedding, which in effect means
    # <PAD> has all-zero embedding, so here we explicitly set <PAD>'s embedding
    # to all-zero.
    tgt_embedding = tf.concat(
        [tf.zeros(shape=[1, src_word_embedder.dim]),
         src_word_embedder.embedding[1:, :]],
        axis=0)
    tgt_embedder = tx.modules.WordEmbedder(tgt_embedding)
    tgt_word_embeds = tgt_embedder(decoder_input)
    tgt_word_embeds = tgt_word_embeds * config_model.hidden_dim ** 0.5

    tgt_seq_len = tf.ones([batch_size], tf.int32) * tf.shape(decoder_input)[1]
    tgt_pos_embeds = pos_embedder(sequence_length=tgt_seq_len)

    tgt_input_embedding = tgt_word_embeds + tgt_pos_embeds

    _output_w = tf.transpose(tgt_embedder.embedding, (1, 0))

    decoder = TransformerDecoder(vocab_size=vocab_size,
                                 output_layer=_output_w,
                                 hparams=config_model.decoder)
    # For training
    outputs = decoder(
        memory=encoder_output,
        memory_sequence_length=encoder_input_length,
        inputs=tgt_input_embedding,
        decoding_strategy='train_greedy',
        mode=tf.estimator.ModeKeys.TRAIN
    )

    mle_loss = transformer_utils.smoothing_cross_entropy(
        outputs.logits, labels, vocab_size, config_model.loss_label_confidence)
    mle_loss = tf.reduce_sum(mle_loss * is_target) / tf.reduce_sum(is_target)

    train_op = tx.core.get_train_op(
        mle_loss,
        learning_rate=learning_rate,
        global_step=global_step,
        hparams=config_model.opt)

    tf.summary.scalar('lr', learning_rate)
    tf.summary.scalar('mle_loss', mle_loss)
    summary_merged = tf.summary.merge_all()

    # For inference (beam-search)
    start_tokens = tf.fill([batch_size], bos_token_id)

    def _embedding_fn(x, y):
        x_w_embed = tgt_embedder(x)
        y_p_embed = pos_embedder(y)
        return x_w_embed * config_model.hidden_dim ** 0.5 + y_p_embed

    predictions = decoder(
        memory=encoder_output,
        memory_sequence_length=encoder_input_length,
        beam_width=beam_width,
        length_penalty=config_model.length_penalty,
        start_tokens=start_tokens,
        end_token=eos_token_id,
        embedding=_embedding_fn,
        max_decoding_length=config_data.max_decoding_length,
        mode=tf.estimator.ModeKeys.PREDICT)
    # Uses the best sample by beam search
    beam_search_ids = predictions['sample_id'][:, :, 0]

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
            sources, targets = zip(*eval_data[i:i + bsize])
            x_block = data_utils.source_pad_concat_convert(sources)
            feed_dict = {
                encoder_input: x_block,
                tx.global_mode(): tf.estimator.ModeKeys.EVAL,
            }
            fetches = {
                'beam_search_ids': beam_search_ids,
            }
            fetches_ = sess.run(fetches, feed_dict=feed_dict)

            hypotheses.extend(h.tolist() for h in fetches_['beam_search_ids'])
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
                hwords, rwords, fname, mode='s',
                src_fname_suffix='hyp', tgt_fname_suffix='ref')
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
            in_arrays = data_utils.seq2seq_pad_concat_convert(train_batch)
            feed_dict = {
                encoder_input: in_arrays[0],
                decoder_input: in_arrays[1],
                labels: in_arrays[2],
                learning_rate: utils.get_lr(step, config_model.lr)
            }
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

    # Run the graph
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        smry_writer = tf.summary.FileWriter(FLAGS.model_dir, graph=sess.graph)

        if FLAGS.run_mode == 'train_and_evaluate':
            logger.info('Begin running with train_and_evaluate mode')

            if tf.train.latest_checkpoint(FLAGS.model_dir) is not None:
                logger.info('Restore latest checkpoint in %s' % FLAGS.model_dir)
                saver.restore(sess, tf.train.latest_checkpoint(FLAGS.model_dir))

            step = 0
            for epoch in range(config_data.max_train_epoch):
                step = _train_epoch(sess, epoch, step, smry_writer)

        elif FLAGS.run_mode == 'test':
            logger.info('Begin running with test mode')

            logger.info('Restore latest checkpoint in %s' % FLAGS.model_dir)
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.model_dir))

            _eval_epoch(sess, 0, mode='test')

        else:
            raise ValueError('Unknown mode: {}'.format(FLAGS.run_mode))


if __name__ == '__main__':
    main()
