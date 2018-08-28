"""
Example pipeline. This is a minimal example of transformer model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import random
import logging
import codecs
import os

import tensorflow as tf
import texar as tx
from texar.modules import TransformerEncoder, TransformerDecoder
from texar.utils import transformer_utils
import config_model
import bleu_tool
from torchtext import data
import utils.data_reader
from utils.data_writer import write_words
from utils.helpers import set_random_seed, batch_size_fn, adjust_lr
from texar.utils.shapes import shape_list

flags = tf.flags
flags.DEFINE_string("config_model", "config_model", "The model config.")
flags.DEFINE_string("config_data", "config_iwslt14", "The dataset config.")
flags.DEFINE_string("run_mode", "train_and_evaluate",
    """choose between train_and_evaluate and test.""")
flags.DEFINE_string("log_dir", "",
    "The path to save the trained model and tensorflow logging.")
flags = flags.FLAGS

config_model = importlib.import_module(FLAGS.config_model)
config_data = importlib.import_module(FLAGS.config_data)

if __name__ == "__main__":
    hparams = config_model.load_hyperparams()
    encoder_hparams, decoder_hparams, opt_hparams, loss_hparams = \
        config_model.encoder_hparams, config_model.decoder_hparams, \
        config_model.opt_hparams, config_model.loss_hparams
    train_data, dev_data, test_data = utils.data_reader.load_data_numpy(\
        config_data.input_dir, args.filename_prefix)
    set_random_seed(config_model.random_seed)
    beam_width = config_model.beam_width
    bos_idx, eos_idx = 1, 2
    # configure the logging module
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logging_file = os.path.join(flags.log_dir, 'logging.txt')
    print('logging file is saved in :{}'.format(logging_file))
    fh = logging.FileHandler(logging_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter('%(asctime)s:%(levelname)s:%(message)s'))
    logger.addHandler(fh)
    with open(args.vocab_file, 'rb') as f: args.id2w = pickle.load(f)
    args.n_vocab = len(args.id2w)
    logger.info('begin logging')

    encoder_input = tf.placeholder(tf.int64, shape=(None, None))
    decoder_input = tf.placeholder(tf.int64, shape=(None, None))
    labels = tf.placeholder(tf.int64, shape=(None, None))
    global_step = tf.Variable(0, dtype=tf.int64, trainable=False)
    global_step_py, best_score, best_epoch = 0, 0, -1
    learning_rate = tf.placeholder(tf.float64, shape=(), name='lr')
    istarget = tf.to_float(tf.not_equal(labels, 0))
    word_embedder = tx.modules.WordEmbedder(
        vocab_size=args.n_vocab,
        hparams=args.word_embedding_hparams,
    )
    encoder = TransformerEncoder(
        hparams=encoder_hparams)
    src_inputs_padding = tf.to_float(tf.equal(encoder_input, 0))
    src_inputs_embedding = tf.multiply(
        word_embedder(encoder_input),
        tf.expand_dims(1 - src_inputs_padding, 2)
    )
    src_lens = tf.reduce_sum(1 - src_inputs_padding, axis=1)
    encoder_output = encoder(inputs=src_inputs_embedding, sequence_length=src_lens)

    # In the decoder, share the word embeddings with projection layer.
    tgt_embeds = tf.concat([
        tf.zeros(shape=[1, encoder_hparams['dim']]),
        word_embedder._embedding[1:, :]], 0
    )
    decoder = TransformerDecoder(
        embedding=tgt_embeds,
        hparams=decoder_hparams)
    tgt_inputs_padding = tf.to_float(tf.equal(decoder_input, 0))
    tgt_lens = tf.reduce_sum(1 - tgt_inputs_padding, axis=1)
    tgt_inputs_embedding = tf.multiply(
        word_embedder(decoder_input),
        tf.expand_dims(1 - tgt_inputs_padding, 2)
    )
    batch_size = shape_list(encoder_output)[0]
    start_tokens = tf.fill([batch_size], 1)
    output = decoder(
        memory=encoder_output,
        memory_sequence_length=src_lens,
        memory_attention_bias=None,
        inputs=tgt_inputs_embedding,
        decoding_strategy='train_greedy',
        mode=tf.estimator.ModeKeys.TRAIN
    )
    logits, preds = output.logits, output.sample_id
    predictions = decoder(
        memory=encoder_output,
        memory_sequence_length=src_lens,
        memory_attention_bias=None,
        inputs=None,
        decoding_strategy='infer_greedy',
        beam_width=beam_width,
        start_tokens=start_tokens,
        end_token=2,
        max_decoding_length=config_data.max_decode_len,
        mode=tf.estimator.ModeKeys.PREDICT
    )
    if beam_width <= 1:
        infered_ids = predictions[0].sample_id #predictions[0] is the OutputTuple
    else:
        infered_ids = predictions['sample_id'][:, :, 0]

    mle_loss = transformer_utils.smoothing_cross_entropy(logits, \
        labels, args.n_vocab, loss_hparams['label_confidence'])
    mle_loss = tf.reduce_sum(mle_loss * istarget) / tf.reduce_sum(istarget)
    tf.summary.scalar('mle_loss', mle_loss)
    acc = tf.reduce_sum(
        tf.to_float(tf.equal(tf.to_int64(preds), labels)) * istarget) \
        / tf.to_float(tf.reduce_sum(istarget))
    tf.summary.scalar('acc', acc)
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        beta1=opt_hparams['Adam_beta1'],
        beta2=opt_hparams['Adam_beta2'],
        epsilon=opt_hparams['Adam_epsilon'],
    )
    train_op = optimizer.minimize(mle_loss, global_step)
    tf.summary.scalar('lr', learning_rate)
    merged = tf.summary.merge_all()
    eval_saver = tf.train.Saver(max_to_keep=5)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    train_finished = False

    def _eval_epoch(cur_sess, epoch):
        references, hypotheses = [], []
        global best_score, best_epoch
        for i in range(0, len(dev_data), args.test_batch_size):
            sources, targets = zip(*dev_data[i:i+args.test_batch_size])
            references.extend(t.tolist() for t in targets)
            x_block = utils.data_reader.source_pad_concat_convert(sources)
            _feed_dict = {
                encoder_input: x_block,
                tx.global_mode(): tf.estimator.ModeKeys.EVAL,
            }
            fetches = {
                'predictions': infered_ids,
            }
            _fetches = sess.run(fetches, feed_dict=_feed_dict)
            hypotheses.extend(h.tolist() for h in _fetches['predictions'])
        outputs_tmp_filename = flags.log_dir + \
            'eval.output'.format(\
            epoch, args.beam_width, args.alpha)
        refer_tmp_filename = os.path.join(flags.log_dir, 'eval.refer')
        with codecs.open(outputs_tmp_filename, 'w+', 'utf-8') as tmpfile, \
            codecs.open(refer_tmp_filename, 'w+', 'utf-8') as tmpref:
            for hyp, tgt in zip(hypotheses, references):
                if eos_idx in hyp:
                    hyp = hyp[:hyp.index(eos_idx)]
                if eos_idx in tgt:
                    tgt = tgt[:tgt.index(eos_idx)]
                tmpfile.write(' '.join([str(i) for i in hyp]) + '\n')
                tmpref.write(' '.join([str(i) for i in tgt]) + '\n')
        eval_bleu = float(100 * bleu_tool.bleu_wrapper(\
            refer_tmp_filename, outputs_tmp_filename, case_sensitive=True))
        logger.info('eval_bleu %f in epoch %d)' % (eval_bleu, epoch))
        if eval_bleu > best_score:
            logger.info('%s epoch, highest bleu %s',epoch, eval_bleu)
            model_path = flags.log_dir + 'my-model.ckpt'
            logger.info('saveing model in %s', model_path)
            best_score, best_epoch = eval_bleu, epoch
            eval_saver.save(sess, model_path)

    def _train_epoch(cur_sess, cur_epoch):
        global train_data, train_finished, global_step_py
        random.shuffle(train_data)
        train_iter = data.iterator.pool(
            train_data,
            args.wbatchsize,
            key=lambda x: (len(x[0]), len(x[1])),
            batch_size_fn=batch_size_fn,
            random_shuffler=
            data.iterator.RandomShuffler()
        )
        for num_steps, train_batch in enumerate(train_iter):
            in_arrays = utils.data_reader.seq2seq_pad_concat_convert(\
                train_batch, -1)
            _feed_dict = {
                encoder_input: in_arrays[0],
                decoder_input: in_arrays[1],
                labels: in_arrays[2],
                tx.global_mode():tf.estimator.ModeKeys.TRAIN,
                learning_rate: adjust_lr(global_step_py, opt_hparams)
            }
            fetches = {
                'target': labels,
                'step': global_step,
                'train_op': train_op,
                'mgd': merged,
                'loss': mle_loss,
            }
            _fetches = sess.run(fetches, feed_dict=_feed_dict)
            global_step_py, loss, mgd, target = \
                _fetches['step'], _fetches['loss'], _fetches['mgd'], \
                _fetches['target']
            if global_step_py % 500 == 0:
                logger.info('step:%s targets:%s loss:%s', \
                    global_step_py, target.shape, loss)
            writer.add_summary(mgd, global_step=global_step_py)
            if global_step_py == config_data.max_training_steps:
                print('reach max training step, loss:{}'.format(loss))
                train_finished = True
            if global_step_py % config_data.eval_steps == 0:
                _eval_epoch(cur_sess, cur_epoch)

    def _test_epoch(cur_sess):
        references, hypotheses, rwords, hwords = [], [], [], []
        for i in range(0, len(test_data), args.test_batch_size):
            sources, targets = \
                zip(*test_data[i: i + args.test_batch_size])
            references.extend(t.tolist() for t in targets)
            x_block = utils.data_reader.source_pad_concat_convert(sources)
            _feed_dict = {
                encoder_input: x_block,
                tx.global_mode(): tf.estimator.ModeKeys.EVAL,
            }
            fetches = {
                'predictions': infered_ids,
            }
            _fetches = sess.run(fetches, feed_dict=_feed_dict)
            hypotheses.extend(h.tolist() for h in _fetches['predictions'])
        for refer, hypo in zip(references, hypotheses):
            if eos_idx in hypo:
                hypo = hypo[:hypo.index(eos_idx)]
            rwords.append([args.id2w[y] for y in refer])
            hwords.append([args.id2w[y] for y in hypo])
        outputs_tmp_filename = flags.log_dir + \
            'test.output'.format(\
            cur_mname)
        refer_tmp_filename = flags.log_dir + 'test.refer'
        write_words(hwords, outputs_tmp_filename)
        write_words(rwords, refer_tmp_filename)
        logger.info('test finished. The output is in %s' % \
            (outputs_tmp_filename))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())
        writer = tf.summary.FileWriter(flags.log_dir, graph=sess.graph)
        if args.mode == 'train_and_evaluate':
            for epoch in range(args.start_epoch, config_data.max_train_epoch):
                _train_epoch(sess, epoch)
        elif args.mode == 'test':
            cur_mname = tf.train.latest_checkpoint(flags.log_dir).split('/')[-1]
            eval_saver.restore(sess, tf.train.latest_checkpoint(flags.log_dir))
            _test_epoch(sess)
