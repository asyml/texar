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
from importlib import reload

import tensorflow as tf
import texar as tx
from texar.modules import TransformerEncoder, TransformerDecoder
from texar.utils import transformer_utils
import hyperparams
import bleu_tool
from torchtext import data
import utils.data_reader
from utils.helpers import set_random_seed, batch_size_fn, adjust_lr

if __name__ == "__main__":
    hparams = hyperparams.load_hyperparams()
    encoder_hparams, decoder_hparams, opt_hparams, loss_hparams, args = \
        hparams['encoder_hparams'], hparams['decoder_hparams'], \
        hparams['opt_hparams'], hparams['loss_hparams'], hparams['args']
    set_random_seed(args.random_seed)

    logging.shutdown() # TODO(zhiting): ?
    reload(logging)
    logging_file = os.path.join(args.log_dir, 'logging.txt')
    print('logging file is saved in :{}'.format(logging_file))
    logging.basicConfig(filename=logging_file, \
        format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
    train_data, dev_data, test_data = utils.data_reader.load_data_numpy(\
        args.input, args.filename_prefix)
    with open(args.vocab_file, 'rb') as f:
        args.id2w = pickle.load(f)
    args.n_vocab = len(args.id2w)

    encoder_input = tf.placeholder(tf.int64, shape=(None, None))
    decoder_input = tf.placeholder(tf.int64, shape=(None, None))
    labels = tf.placeholder(tf.int64, shape=(None, None))
    global_step = tf.Variable(0, dtype=tf.int64, trainable=False)
    global_step_py = 0
    learning_rate  = tf.placeholder(tf.float64, shape=(), name='lr')
    istarget = tf.to_float(tf.not_equal(labels, 0))
    word_embedder = tx.modules.WordEmbedder(
        vocab_size=args.n_vocab,
        hparams=args.word_embedding_hparams,
    )
    encoder = TransformerEncoder(hparams=encoder_hparams)
    inputs_padding = tf.to_float(tf.equal(encoder_input, 0))
    enc = word_embedder(encoder_input)
    encoder_output, encoder_decoder_attention_bias = \
        encoder(enc, inputs_padding)
    decoder = TransformerDecoder(
        embedding=word_embedder._embedding,
        hparams=decoder_hparams)
    logits, preds=decoder(
        encoder_output,
        encoder_decoder_attention_bias,
        decoder_input,
        mode=tf.estimator.ModeKeys.TRAIN)
    predictions = decoder(
        encoder_output,
        encoder_decoder_attention_bias,
        decoder_input=None,
        mode=tf.estimator.ModeKeys.PREDICT)
    mle_loss = transformer_utils.smoothing_cross_entropy(logits,
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
    config = tf.ConfigProto(
        allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    best_score, best_epoch = 0, -1
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
                    'predictions': predictions['sampled_ids'][:, 0, :],
            }
            _fetches = sess.run(fetches, feed_dict=_feed_dict)
            hypotheses.extend(h.tolist() for h in _fetches['predictions'])
        # Threshold Global Steps to save the model
        outputs_tmp_filename = args.log_dir + \
            'my_model_epoch{}.beam{}alpha{}.outputs.tmp'.format(\
            epoch , args.beam_width, args.alpha)
        refer_tmp_filename = os.path.join(args.log_dir, 'val_refer.tmp')
        with codecs.open(outputs_tmp_filename, 'w+', 'utf-8') as tmpfile, \
                codecs.open(refer_tmp_filename, 'w+', 'utf-8') as tmpref:
            for hyp, tgt in zip(hypotheses, references):
                if decoder._hparams.eos_id in hyp:
                    hyp = hyp[:hyp.index(decoder._hparams.eos_id)]
                if decoder._hparams.eos_id in tgt:
                    tgt = tgt[:tgt.index(decoder._hparams.eos_id)]
                str_hyp = [str(i) for i in hyp]
                str_tgt = [str(i) for i in tgt]
                tmpfile.write(' '.join(str_hyp) + '\n')
                tmpref.write(' '.join(str_tgt) + '\n')
        eval_bleu = float(100 * bleu_tool.bleu_wrapper(\
            refer_tmp_filename, outputs_tmp_filename, case_sensitive=True))
        logging.info('eval_bleu %f in epoch %d)' % (eval_bleu, epoch))
        if eval_bleu > best_score:
            logging.info('the %s epoch, highest bleu %s', \
                epoch, eval_bleu)
            best_score = eval_bleu
            best_epoch = epoch
            eval_saver.save(sess,
                args.log_dir + 'my-model-highest_bleu.ckpt')

    def _train_epoch(cur_sess, cur_epoch):
        global train_data, train_finished, global_step_py
        random.shuffle(train_data)
        train_iter = data.iterator.pool(train_data,
                                        args.wbatchsize,
                                        key=lambda x: (len(x[0]), len(x[1])),
                                        batch_size_fn=batch_size_fn,
                                        random_shuffler=
                                        data.iterator.RandomShuffler())
        for num_steps, train_batch in enumerate(train_iter):
            in_arrays = utils.data_reader.seq2seq_pad_concat_convert(train_batch, -1)
            _feed_dict = {
                encoder_input: in_arrays[0],
                decoder_input: in_arrays[1],
                labels: in_arrays[2],
                tx.global_mode():tf.estimator.ModeKeys.TRAIN,
                learning_rate: adjust_lr(global_step_py, opt_hparams)
            }
            fetches = {
                'target': labels,
                'logits': logits,
                'predictions': preds,
                'step': global_step,
                'train_op': train_op,
                'mgd': merged,
            }
            _fetches = sess.run(fetches, feed_dict=_feed_dict)
            global_step_py, loss, mgd, source, target = _fetches['step'], \
                _fetches['train_op'], _fetches['mgd'], \
                _fetches['source'], _fetches['target']
            if global_step_py % 500 == 0:
                logging.info('step:%s source:%s targets:%s loss:%s', \
                    global_step_py, source.shape, target.shape, loss)
            writer.add_summary(mgd, global_step=global_step_py)
            if global_step_py == opt_hparams['max_training_steps']:
                print('reach max training step, loss:{}'.format(loss))
                train_finished = True
            if global_step_py % args.eval_steps == 0:
                _eval_epoch(cur_sess, cur_epoch)

    def _test_epoch(cur_sess):
        references, hypotheses, rwords, hwords = [], [], [], []
        for i in range(0, len(test_data), args.test_batch_size):
            sources, targets = zip(*test_data[i: i+args.test_batch_size])
            references.extend(t.tolist() for t in targets)
            x_block = utils.data_reader.source_pad_concat_convert(sources)
            _feed_dict = {
                encoder_input: x_block,
                tx.global_mode(): tf.estimator.ModeKeys.EVAL,
            }
            fetches ={
                    'predictions': predictions['sampled_ids'][:, 0, :],
            }
            _fetches = sess.run(fetches, feed_dict=_feed_dict)
            hypotheses.extend(h.tolist() for h in _fetches['predictions'])
        for refer, hypo in zip(references, hypotheses):
            if 2 in hypo:
                hypo = hypo[:hypo.index(2)]
            rwords.append([args.id2w[y] for y in refer])
            hwords.append([args.id2w[y] for y in hypo])
        outputs_tmp_filename = args.log_dir + \
            '{}.test.beam{}alpha{}.outputs.decodes'.format(\
            cur_mname, args.beam_width, args.alpha)
        refer_tmp_filename = args.log_dir + 'test_reference.tmp'
        with codecs.open(outputs_tmp_filename, 'w+', 'utf-8') as tmpfile, \
                codecs.open(refer_tmp_filename, 'w+', 'utf-8') as tmpref:
            for hyp, tgt in zip(hwords, rwords):
                tmpfile.write(' '.join(hyp) + '\n')
                tmpref.write(' '.join(tgt) + '\n')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())
        var_list = tf.trainable_variables()
        with open(args.log_dir + 'var.list', 'w+') as outfile:
            for var in var_list:
                outfile.write('var:{} shape:{} dtype:{}\n'.format(\
                    var.name, var.shape, var.dtype))
        writer = tf.summary.FileWriter(args.log_dir, graph=sess.graph)
        if args.mode == 'train_and_evaluate':
            for epoch in range(args.start_epoch, args.epoch):
                _train_epoch(sess, epoch)
        elif args.mode == 'test':
            cur_mname = tf.train.latest_checkpoint(args.log_dir).split('/')[-1]
            eval_saver.restore(sess,
                tf.train.latest_checkpoint(args.log_dir))
            _test_epoch(sess)
