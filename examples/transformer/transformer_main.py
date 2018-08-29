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
import tensorflow as tf
import texar as tx
from texar.modules import TransformerEncoder, TransformerDecoder
from texar.utils import transformer_utils

from utils import data_utils
from utils  import utils
from bleu_tool import bleu_wrapper

# pylint: disable=invalid-name, too-many-locals

flags = tf.flags

flags.DEFINE_string("config_model", "config_model", "The model config.")
flags.DEFINE_string("config_data", "config_iwslt14", "The dataset config.")
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
    bos_token_id, eos_token_id = 1, 2

    beam_width = config_model.beam_width

    # Create logging
    tx.utils.maybe_create_dir(FLAGS.model_dir)
    logging_file = os.path.join(FLAGS.model_dir, 'logging.txt')
    logger = utils.get_logger(logging_file)
    logger.info('logging file is saved in: %s', logging_file)

    # Build model graph
    encoder_input = tf.placeholder(tf.int64, shape=(None, None))
    decoder_input = tf.placeholder(tf.int64, shape=(None, None))
    # (text sequence length excluding padding)
    encoder_input_length = tf.reduce_sum(
        1 - tf.to_int32(tf.equal(encoder_input, 0)), axis=1)
    decoder_input_length = tf.reduce_sum(
        1 - tf.to_int32(tf.equal(decoder_input, 0)), axis=1)

    labels = tf.placeholder(tf.int64, shape=(None, None))
    is_target = tf.to_float(tf.not_equal(labels, 0))

    global_step = tf.Variable(0, dtype=tf.int64, trainable=False)
    learning_rate = tf.placeholder(tf.float64, shape=(), name='lr')

    embedder = tx.modules.WordEmbedder(
        vocab_size=vocab_size, hparams=config_model.emb)
    encoder = TransformerEncoder(hparams=config_model.encoder)

    encoder_output = encoder(inputs=embedder(encoder_input),
                             sequence_length=encoder_input_length)

    # The decoder ties the input word embedding with the output logit layer.
    # As the decoder masks out <PAD>'s embedding, which in effect means
    # <PAD> has all-zero embedding, so here we explicitly set <PAD>'s embedding
    # to all-zero.
    tgt_embedding = tf.concat(
        [tf.zeros(shape=[1, embedder.dim]), embedder.embedding[1:, :]], axis=0)
    decoder = TransformerDecoder(embedding=tgt_embedding,
                                 hparams=config_model.decoder)
    outputs = decoder(
        memory=encoder_output,
        memory_sequence_length=encoder_input_length,
        inputs=embedder(decoder_input),
        sequence_length=decoder_input_length,
        decoding_strategy='train_greedy',
        mode=tf.estimator.ModeKeys.TRAIN
    )

    mle_loss = transformer_utils.smoothing_cross_entropy(
        outputs.logits, labels, vocab_size, config_model.loss_label_confidence)
    mle_loss = tf.reduce_sum(mle_loss * is_target) / tf.reduce_sum(is_target)

    train_op = tx.core.get_train_op(
        mle_loss, learning_rate=learning_rate, hparams=config_model.opt)

    #optimizer = tf.train.AdamOptimizer(
    #    learning_rate=learning_rate,
    #    beta1=config_model.opt['Adam_beta1'],
    #    beta2=config_model.opt['Adam_beta2'],
    #    epsilon=config_model.opt['Adam_epsilon'],
    #)
    #train_op = optimizer.minimize(mle_loss, global_step)

    tf.summary.scalar('lr', learning_rate)
    tf.summary.scalar('mle_loss', mle_loss)
    summary_merged = tf.summary.merge_all()

    saver = tf.train.Saver(max_to_keep=5)

    start_tokens = tf.fill([tx.utils.get_batch_size(encoder_input)],
                           bos_token_id)
    predictions = decoder(
        memory=encoder_output,
        memory_sequence_length=encoder_input_length,
        decoding_strategy='infer_greedy',
        beam_width=beam_width,
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


    best_results = {'score': 0, 'epoch': -1}

    def _eval_epoch(sess, epoch):
        references, hypotheses = [], []
        for i in range(0, len(dev_data), config_data.test_batch_size):
            sources, targets = zip(*dev_data[i:i+config_data.test_batch_size])
            x_block = data_utils.source_pad_concat_convert(sources)

            feed_dict = {
                encoder_input: x_block,
                tx.global_mode(): tf.estimator.ModeKeys.EVAL,
            }
            fetches = {
                'inferred_ids': inferred_ids,
            }

            fetches_ = sess.run(fetches, feed_dict=feed_dict)

            hypotheses.extend(hyp.tolist() for hyp in fetches_['inferred_ids'])
            references.extend(ref.tolist() for ref in targets)

        hypotheses = utils.list_strip_eos(hypotheses, eos_token_id)
        hypotheses = tx.utils.str_join(hypotheses)
        references = utils.list_strip_eos(references, eos_token_id)
        references = tx.utils.str_join(references)

        fname = os.path.join(FLAGS.model_dir, 'eval')
        hyp_fn, ref_fn = tx.utils.write_paired_text( # TODO(zhiting): append=True ?
            hypotheses, references, fname, append=True, mode='s')

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


    def _train_epoch(sess, epoch, smry_writer):
        random.shuffle(train_data)
        train_iter = data.iterator.pool(
            train_data,
            config_data.batch_size,
            key=lambda x: (len(x[0]), len(x[1])),
            batch_size_fn=utils.batch_size_fn,
            random_shuffler=data.iterator.RandomShuffler())

        step = 0
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

            step = fetches_['step']

            if step % config_data.display_steps == 0:
                logger.info('step: %d, loss: %.4f', step, fetches_['loss'])
                print('step: %d, loss: %.4f' % (step, fetches_['loss']))
                smry_writer.add_summary(fetches_['smry'], global_step=step)

            if step % config_data.eval_steps == 0:
                _eval_epoch(sess, epoch)


    # TODO(zhiting): delete the func
    def _test_epoch(sess, ckpt_fn):
        references, hypotheses, rwords, hwords = [], [], [], []
        for i in range(0, len(test_data), config_data.test_batch_size):
            sources, targets = \
                zip(*test_data[i: i + config_data.test_batch_size])
            references.extend(t.tolist() for t in targets)
            x_block = data_utils.source_pad_concat_convert(sources)
            feed_dict = {
                encoder_input: x_block,
                tx.global_mode(): tf.estimator.ModeKeys.EVAL,
            }
            fetches = {
                'predictions': inferred_ids,
            }
            fetches_ = sess.run(fetches, feed_dict=feed_dict)
            hypotheses.extend(h.tolist() for h in fetches_['predictions'])
        for refer, hypo in zip(references, hypotheses):
            if eos_token_id in hypo:
                hypo = hypo[:hypo.index(eos_token_id)]
            rwords.append([id2w[y] for y in refer])
            hwords.append([id2w[y] for y in hypo])
        outputs_tmp_filename = FLAGS.model_dir + \
            'test.output'.format(\
            ckpt_fn) # TODO(zhiting): error: `Too many arguments for format string`
        refer_tmp_filename = FLAGS.model_dir + 'test.refer'
        data_utils.write_words(hwords, outputs_tmp_filename)
        data_utils.write_words(rwords, refer_tmp_filename)
        logger.info('test finished. The output is in %s' % \
            (outputs_tmp_filename))

    # Run the graph
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        smry_writer = tf.summary.FileWriter(FLAGS.model_dir, graph=sess.graph)

        if FLAGS.run_mode == 'train_and_evaluate':
            for epoch in range(config_data.max_train_epoch):
                _train_epoch(sess, epoch, smry_writer)

        elif FLAGS.run_mode == 'test':
            ckpt_fn = tf.train.latest_checkpoint(FLAGS.model_dir).split('/')[-1]
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.model_dir))
            _test_epoch(sess, ckpt_fn)


if __name__ == '__main__':
    main()
