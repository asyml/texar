"""
Example pipeline. This is a minimal example of transfomrer model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import random
import logging
#import codecs
import codecs
import os
from importlib import reload
import numpy as np
from matplotlib import pyplot as plt
plt.switch_backend('agg')

# pylint: disable=wrong-import-position
import tensorflow as tf
import texar as tx
import itertools
# pylint: disable=invalid-name, no-name-in-module
from texar.modules import TransformerEncoder, TransformerDecoder
from texar.utils import transformer_utils
import hyperparams
import bleu_tool
from torchtext import data
import utils
global max_src_in_batch, max_tgt_in_batch

def batch_size_fn(new, count, sofar):
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new[0]) + 2)
    max_tgt_in_batch = max(max_tgt_in_batch, len(new[1]) + 1)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)

if __name__ == "__main__":
    tf.set_random_seed(1234)
    np.random.seed(1234)
    random.seed(1234)
    hparams = hyperparams.load_hyperparams()
    encoder_hparams, decoder_hparams, opt_hparams, loss_hparams, args = \
        hparams['encoder_hparams'], hparams['decoder_hparams'], \
        hparams['opt_hparams'], hparams['loss_hparams'], hparams['args']

    logging.shutdown()
    reload(logging)
    logging_file = os.path.join(args.log_dir, 'logging.txt')
    print('logging file is saved in :{}'.format(logging_file))
    logging.basicConfig(filename=logging_file, \
        format='%(asctime)s:%(levelname)s:%(message)s',\
        level=logging.INFO)
    logging.info('begin logging, new running')
    # Construct the databas

    train_data = np.load(os.path.join(args.input, args.data + '.train.npy'))
    train_data = train_data.tolist()
    dev_data = np.load(os.path.join(args.input, args.data + '.valid.npy'))
    dev_data = dev_data.tolist()
    test_data = np.load(os.path.join(args.input, args.data + '.test.npy'))
    test_data = test_data.tolist()

    with open(os.path.join(args.input, args.data + '.vocab.pickle'),
        'rb') as f:
        id2w = pickle.load(f)

    args.id2w = id2w
    args.n_vocab = len(id2w)

    encoder_input = tf.placeholder(tf.int64, shape=(None, None))
    decoder_input = tf.placeholder(tf.int64, shape=(None, None))
    labels = tf.placeholder(tf.int64, shape=(None, None))
    istarget = tf.to_float(tf.not_equal(labels, 0))

    WordEmbedder = tx.modules.WordEmbedder(
        vocab_size=args.n_vocab,
        hparams=args.word_embedding_hparams,
    )
    encoder = TransformerEncoder(
        embedding=WordEmbedder._embedding,
        hparams=encoder_hparams)
    encoder_output, encoder_decoder_attention_bias = \
        encoder(encoder_input)
    decoder = TransformerDecoder(
        embedding=encoder._embedding,
        hparams=decoder_hparams)
    logits, preds=decoder(
        decoder_input,
        encoder_output,
        encoder_decoder_attention_bias)
    predictions = decoder.dynamic_decode(
        encoder_output,
        encoder_decoder_attention_bias,
    )
    mle_loss = transformer_utils.smoothing_cross_entropy(
        logits,
        labels,
        args.n_vocab,
        loss_hparams['label_confidence'],
    )

    mle_loss = tf.reduce_sum(mle_loss * istarget) / tf.reduce_sum(istarget)
    tf.summary.scalar('mle_loss', mle_loss)
    acc = tf.reduce_sum(
        tf.to_float(tf.equal(tf.to_int64(preds), labels)) * istarget) \
        / tf.to_float(tf.reduce_sum(istarget))
    tf.summary.scalar('acc', acc)
    global_step = tf.Variable(0, dtype=tf.int64, trainable=False)
    fstep = tf.to_float(global_step)
    if opt_hparams['learning_rate_schedule'] == 'static':
        learning_rate = 1e-3
    else:
        learning_rate = opt_hparams['lr_constant'] \
            * tf.minimum(1.0, (fstep / opt_hparams['warmup_steps'])) \
            * tf.rsqrt(tf.maximum(fstep, opt_hparams['warmup_steps'])) \
            * args.hidden_dim**-0.5
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
    src_data, tgt_data = list(zip(*train_data))
    total_src_words = len(list(itertools.chain.from_iterable(src_data)))
    total_tgt_words = len(list(itertools.chain.from_iterable(tgt_data)))
    iter_per_epoch = (total_src_words + total_tgt_words) // (2 * args.wbatchsize)
    print('Approximate number of iter/epoch =', iter_per_epoch)
    best_score = 0
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
        eval_writer = tf.summary.FileWriter(os.path.join(args.log_dir, \
            'eval/'), graph=sess.graph)
        if args.mode == 'train_and_evaluate':
            for epoch in range(args.start_epoch, args.epoch):
                random.shuffle(train_data)
                train_iter = data.iterator.pool(train_data,
                                                args.wbatchsize,
                                                key=lambda x: (len(x[0]), len(x[1])),
                                                batch_size_fn=batch_size_fn,
                                                random_shuffler=
                                                data.iterator.RandomShuffler())
                report_stats = utils.Statistics()
                train_stats = utils.Statistics()
                for num_steps, train_batch in enumerate(train_iter):
                    src_iter = list(zip(*train_batch))[0]
                    src_words = len(list(itertools.chain.from_iterable(src_iter)))
                    report_stats.n_src_words += src_words
                    train_stats.n_src_words += src_words
                    in_arrays = utils.seq2seq_pad_concat_convert(train_batch, -1)
                    _feed_dict = {
                        encoder_input: in_arrays[0],
                        decoder_input: in_arrays[1],
                        labels: in_arrays[2],
                        tx.global_mode():tf.estimator.ModeKeys.TRAIN,
                    }
                    fetches = {
                        'source': encoder_input,
                        'target': labels,
                        'logits': logits,
                        'predictions': preds,
                        'loss': mle_loss,
                        'step': global_step,
                        'train_op': train_op,
                        'mgd': merged,
                    }
                    _fetches = sess.run(fetches, feed_dict=_feed_dict)
                    step, loss, mgd, source, target = _fetches['step'], _fetches['loss'], \
                        _fetches['mgd'], _fetches['source'], _fetches['target']
                    if step % 100 == 0:
                        logging.info('step:%s source:%s targets:%s loss:%s', \
                            step, source.shape, target.shape, loss)
                    writer.add_summary(mgd, global_step=step)
                    if step == opt_hparams['max_training_steps']:
                        print('reach max steps:{} loss:{}'.format(step, loss))
                    if (step+1) % args.eval_steps == 0:
                        references = []
                        hypotheses = []
                        for i in range(0, len(dev_data), args.test_batch_size):
                            sources, targets = zip(*dev_data[i:i+args.test_batch_size])
                            references.extend(t.tolist() for t in targets)
                            x_block = utils.source_pad_concat_convert(sources)

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
                            epoch, args.beam_width, args.alpha)
                        refer_tmp_filename = os.path.join(args.log_dir, 'eval_reference.tmp')
                        with codecs.open(outputs_tmp_filename, 'w+', 'utf-8') as tmpfile, \
                                codecs.open(refer_tmp_filename, 'w+', 'utf-8') as tmpreffile:
                            for hyp, tgt in zip(hypotheses, references):
                                if 2 in hyp:
                                    hyp = hyp[:hyp.index(2)]
                                if 2 in tgt:
                                    tgt = tgt[:tgt.index(2)]
                                str_hyp = [str(i) for i in hyp]
                                str_tgt = [str(i) for i in tgt]
                                tmpfile.write(' '.join(str_hyp) + '\n')
                                tmpreffile.write(' '.join(str_tgt) + '\n')
                        eval_bleu = float(100 * bleu_tool.bleu_wrapper(\
                            refer_tmp_filename, outputs_tmp_filename, case_sensitive=True))
                        logging.info('eval_bleu %f in epoch %d)' % (eval_bleu, epoch))
                        if step > 8000:
                            if eval_bleu > best_score:
                                logging.info('the %s epoch, highest bleu %s', \
                                    epoch, eval_bleu)
                                best_score = eval_bleu
                                best_epoch = epoch
                                eval_saver.save(sess,
                                    args.log_dir + 'my-model-highest_bleu.ckpt')
        elif args.mode == 'test':
            cur_mname = tf.train.latest_checkpoint(args.log_dir).split('/')[-1]
            eval_saver.restore(sess,
                tf.train.latest_checkpoint(args.log_dir))
            references, hypotheses = [], []
            for i in range(0, len(test_data), args.test_batch_size):
                sources, targets = zip(*test_data[i: i+args.test_batch_size])
                references.extend(t.tolist() for t in targets)
                x_block = utils.source_pad_concat_convert(sources)
                _feed_dict = {
                    encoder_input: x_block,
                    tx.global_mode(): tf.estimator.ModeKeys.EVAL,
                }
                fetches ={
                        'predictions': predictions['sampled_ids'][:, 0, :],
                }
                _fetches = sess.run(fetches, feed_dict=_feed_dict)
                hypotheses.extend(h.tolist() for h in _fetches['predictions'])
            references_words = []
            hypotheses_words = []
            for refer, hypo in zip(references, hypotheses):
                if 2 in hypo:
                    hypo = hypo[:hypo.index(2)]
                references_words.append([args.id2w[y] for y in refer])
                hypotheses_words.append([args.id2w[y] for y in hypo])
            outputs_tmp_filename = args.log_dir + \
                '{}.test.beam{}alpha{}.outputs.decodes'.format(\
                cur_mname, args.beam_width, args.alpha)
            refer_tmp_filename = args.log_dir + 'test_reference.tmp'
            with codecs.open(outputs_tmp_filename, 'w+', 'utf-8') as tmpfile, \
                    codecs.open(refer_tmp_filename, 'w+', 'utf-8') as tmpreffile:
                for hyp, tgt in zip(hypotheses_words, references_words):
                    tmpfile.write(' '.join(hyp) + '\n')
                    tmpreffile.write(' '.join(tgt) + '\n')
