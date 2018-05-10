"""
Example pipeline. This is a minimal example of transfomrer model.
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
import numpy as np
from matplotlib import pyplot as plt
plt.switch_backend('agg')

# pylint: disable=wrong-import-position
import tensorflow as tf
import texar as tx
# pylint: disable=invalid-name, no-name-in-module
from texar.modules import TransformerEncoder, TransformerDecoder
from texar.losses import mle_losses
from texar import context

from hyperparams import train_dataset_hparams, eval_dataset_hparams, \
    test_dataset_hparams, \
    encoder_hparams, decoder_hparams, \
    opt_hparams, loss_hparams, args
import bleu_tool

if __name__ == "__main__":
    logging.shutdown()
    reload(logging)
    logging_file = os.path.join(args.log_dir, 'logging.txt')
    logging.basicConfig(filename=logging_file, \
        format='%(asctime)s:%(levelname)s:%(message)s',\
        level=logging.INFO)
    logging.info('begin logging, new running')
    tf.set_random_seed(1234)
    np.random.seed(1234)
    random.seed(1234)
    # Construct the database
    train_database = tx.data.PairedTextData(train_dataset_hparams)
    eval_database = tx.data.PairedTextData(eval_dataset_hparams)
    test_database = tx.data.PairedTextData(test_dataset_hparams)

    # TODO: uncomment this will cause protected_access warning of pylint
    #logging.info('train_database hparams: %s', train_database._hparams)
    #logging.info('eval_database hparams: %s', eval_database._hparams)
    #logging.info('test_database hparams: %s', test_database._hparams)

    iterator = tx.data.TrainTestDataIterator(train=train_database,
                                             val=eval_database,
                                             test=test_database)
    text_data_batch = iterator.get_next()

    ori_src_text = text_data_batch['source_text_ids']
    ori_tgt_text = text_data_batch['target_text_ids']

    encoder_input = ori_src_text[:, 1:]
    decoder_input = ori_tgt_text[:, :-1]
    labels = ori_tgt_text[:, 1:]
    enc_padding = tf.to_float(tf.equal(encoder_input, 0))
    dec_padding = tf.to_float(tf.equal(decoder_input, 0))
    istarget = tf.to_float(tf.not_equal(labels, 0))

    WordEmbedder = tx.modules.WordEmbedder(
        vocab_size=train_database.source_vocab.size,
        hparams=args.word_embedding_hparams,
    )
    #logging.info('WordEmbedder hparams: %s', WordEmbedder._hparams)

    # TODO(haoran): a better interface to share embedding
    encoder = TransformerEncoder(
        embedding=WordEmbedder._embedding,
        vocab_size=train_database.source_vocab.size,\
        hparams=encoder_hparams)
    #logging.info('encoder hparams: %s', encoder._hparams)
    encoder_output, encoder_decoder_attention_bias = \
        encoder(encoder_input, enc_padding)

    decoder = TransformerDecoder(
        embedding=encoder._embedding,
        hparams=decoder_hparams)
    #logging.info('decode hparams: %s', decoder._hparams)
    logits, preds = decoder(
        decoder_input,
        encoder_output,
        encoder_decoder_attention_bias,
    )
    predictions = decoder.dynamic_decode(
        encoder_output,
        encoder_decoder_attention_bias,
    )

    mle_loss = mle_losses.smoothing_cross_entropy(
        logits,
        labels,
        train_database.target_vocab.size,
        loss_hparams['label_confidence'],
    )
    mle_loss = tf.reduce_sum(mle_loss * istarget) / tf.reduce_sum(istarget)
    tf.summary.scalar('mle_loss', mle_loss)

    acc = tf.reduce_sum(
        tf.to_float(tf.equal(tf.to_int64(preds), labels))*istarget) \
        / tf.to_float((tf.reduce_sum(istarget)))

    tf.summary.scalar('acc', acc)

    global_step = tf.Variable(0, trainable=False)
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

    vocab = train_database.source_vocab
    #graph = tf.get_default_graph()
    #graph.finalize()

    def _train_epochs(session, cur_epoch, sess_writer):
        iterator.switch_to_train_data(session)
        if args.draw_for_debug:
            loss_lists = []
        while True:
            try:
                fetches = {'source': encoder_input,
                           'dec_in': decoder_input,
                           'target': labels,
                           'predict': preds,
                           'train_op': train_op,
                           'step': global_step,
                           'loss': mle_loss,
                           'mgd' :merged}
                feed = {context.global_mode(): tf.estimator.ModeKeys.TRAIN}
                _fetches = session.run(fetches, feed_dict=feed)
                step, source, target, loss, mgd = _fetches['step'], \
                    _fetches['source'], _fetches['target'], \
                    _fetches['loss'], _fetches['mgd']
                if args.draw_for_debug:
                    loss_lists.append(loss)
                if step % 3000 == 0:
                    logging.info('step:%s source:%s targets:%s loss:%s', \
                        step, source.shape, target.shape, loss)

                sess_writer.add_summary(mgd, global_step=step)
                if step == opt_hparams['max_training_steps']:
                    print('reach max steps:{} loss:{}'.format(step, loss))
                    logging.info('reached max training steps')
                    return 'finished'
            except tf.errors.OutOfRangeError:
                break
        logging.info('step:%s loss:%s epoch:%s', step, loss, cur_epoch)
        if args.draw_for_debug:
            plt.figure(figsize=(14, 10))
            plt.plot(loss_lists, '--', linewidth=1, label='loss trend')
            plt.ylabel('training loss')
            plt.xlabel('training steps in one epoch')
            plt.savefig('train_loss_curve.epoch{}.png'.format(cur_epoch))
        print('step:{} loss:{} epoch:{}'.format(step, loss, cur_epoch))
        return 'done'

    def _eval_epoch(cur_sess, cur_epoch, outputs_tmp_filename=None):
        # pylint:disable=too-many-locals
        iterator.switch_to_val_data(cur_sess)
        sources_list, targets_list, hypothesis_list = [], [], []
        eloss = []
        while True:
            try:
                fetches = {
                    'predictions': predictions,
                    'source': ori_src_text,
                    'target': labels,
                    'step': global_step,
                    'mle_loss': mle_loss,
                }
                feed = {context.global_mode(): tf.estimator.ModeKeys.EVAL}
                _fetches = cur_sess.run(fetches, feed_dict=feed)
                sources, sampled_ids, targets = \
                    _fetches['source'].tolist(), \
                    _fetches['predictions']['sampled_ids'][:, 0, :].tolist(), \
                    _fetches['target'].tolist()
                eloss.append(_fetches['mle_loss'])
                if args.verbose:
                    print('cur loss:{}'.format(_fetches['mle_loss']))
                def _id2word_map(id_arrays):
                    return [' '.join([vocab._id_to_token_map_py[i] \
                        for i in sent]) for sent in id_arrays]
                sources, targets, dwords = _id2word_map(sources), \
                                           _id2word_map(targets), \
                                           _id2word_map(sampled_ids)
                for source, target, pred in zip(sources, targets, dwords):
                    source = source.split('<EOS>')[0].strip().split()
                    target = target.split('<EOS>')[0].strip().split()
                    got = pred.split('<EOS>')[0].strip().split()
                    sources_list.append(source)
                    targets_list.append(target)
                    hypothesis_list.append(got)
            except tf.errors.OutOfRangeError:
                break
        outputs_tmp_filename = args.log_dir + \
            'my_model_epoch{}.beam{}alpha{}.outputs.tmp'.format(\
            cur_epoch, args.beam_width, args.alpha)
        refer_tmp_filename = os.path.join(args.log_dir, 'eval_reference.tmp')
        with codecs.open(outputs_tmp_filename, 'w+', 'utf-8') as tmpfile, \
                codecs.open(refer_tmp_filename, 'w+', 'utf-8') as tmpreffile:
            for hyp, tgt in zip(hypothesis_list, targets_list):
                tmpfile.write(' '.join(hyp) + '\n')
                tmpreffile.write(' '.join(tgt) + '\n')
        eval_bleu = float(100 * bleu_tool.bleu_wrapper(\
            refer_tmp_filename, outputs_tmp_filename, case_sensitive=True))
        eloss = float(np.average(np.array(eloss)))
        print('epoch:{} eval_bleu:{} eval_loss:{}'.format(cur_epoch, \
            eval_bleu, eloss))
        if args.save_eval_output:
            with codecs.open(args.log_dir + \
                'my_model_epoch{}.beam{}alpha{}.outputs.bleu{:.3f}'.format(\
                cur_epoch, args.beam_width, args.alpha, eval_bleu), \
                'w+', 'utf-8') as outputfile, codecs.open(args.log_dir + \
                'my_model_epoch{}.beam{}alpha{}.results.bleu{:.3f}'.format(\
                cur_epoch, args.beam_width, args.alpha, eval_bleu), \
                'w+', 'utf-8') as resultfile:
                for src, tgt, hyp in zip(sources_list, targets_list, \
                    hypothesis_list):
                    outputfile.write(' '.join(hyp) + '\n')
                    resultfile.write("- source: " + ' '.join(src) + '\n')
                    resultfile.write("- expected: " + ' '.join(tgt) + '\n')
                    resultfile.write('- got: ' + ' '.join(hyp)+ '\n\n')
        return {'loss': eloss,
                'bleu': eval_bleu
               }

    def _test_epoch(cur_sess, cur_mname):
        # pylint:disable=too-many-locals
        iterator.switch_to_test_data(sess)
        sources_list, targets_list, hypothesis_list = [], [], []
        test_loss, test_bleu = [], 0
        if args.debug:
            fetches = {
                'source': ori_src_text,
                'target': labels,
                'encoder_padding': enc_padding,
                'encoder_embedding': encoder._embedding,
                'encoder_attout': encoder.stack_output,
                'encoder_output': encoder_output,
                'decoder_embedding': decoder._embedding,
                'predictions': predictions,
            }
            feed = {context.global_mode(): tf.estimator.ModeKeys.PREDICT}
            _fetches = cur_sess.run(fetches, feed_dict=feed)
            print('source:{}'.format(_fetches['source']))
            print('target:{}'.format(_fetches['target']))
            print('encoder_padding:{}'.format(_fetches['encoder_padding']))
            print('encoder_embedding:{}'.format(_fetches['encoder_embedding']))
            print('encoder_attout:{}'.format(_fetches['encoder_attout']))
            print('encoder_output:{}'.format(_fetches['encoder_output']))
            print('decoder_embedding:{}'.format(_fetches['decoder_embedding']))
            print('predictions:{}'.format(_fetches['predictions']))
            sources, sampled_ids, targets = \
                _fetches['source'].tolist(), \
                _fetches['predictions']['sampled_ids'][:, 0, :].tolist(), \
                _fetches['target'].tolist()
            exit()

        while True:
            try:
                fetches = {
                    'predictions': predictions,
                    'source': ori_src_text,
                    'target': labels,
                    'step': global_step,
                    'mle_loss': mle_loss,
                    'encoder_output': encoder_output,
                    'decoder_input': decoder_input,
                    'embedding': encoder._embedding,
                    'encoder_decoder_attention_bias': \
                        encoder_decoder_attention_bias,
                    'logits': logits,
                }
                feed = {context.global_mode(): tf.estimator.ModeKeys.PREDICT}
                _fetches = cur_sess.run(fetches, feed_dict=feed)
                sources, sampled_ids, targets = \
                    _fetches['source'].tolist(), \
                    _fetches['predictions']['sampled_ids'][:, 0, :].tolist(), \
                    _fetches['target'].tolist()
                test_loss.append(_fetches['mle_loss'])
                def _id2word_map(id_arrays):
                    return [' '.join([vocab._id_to_token_map_py[i] \
                            for i in sent]) for sent in id_arrays]
                if args.debug:
                    print('source_ids:%s\ntargets_ids:%s\nsampled_ids:%s', \
                        sources, targets, sampled_ids)
                    print('encoder_output:%s %s', \
                        _fetches['encoder_output'].shape, \
                        _fetches['encoder_output'])
                    print('logits:%s %s', _fetches['logits'].shape, \
                        _fetches['logits'])
                    exit()
                sources, targets, dwords = _id2word_map(sources), \
                                           _id2word_map(targets), \
                                            _id2word_map(sampled_ids)
                for source, target, pred in zip(sources, targets, dwords):
                    source = source.split('<EOS>')[0].strip().split()
                    target = target.split('<EOS>')[0].strip().split()
                    got = pred.split('<EOS>')[0].strip().split()
                    sources_list.append(source)
                    targets_list.append(target)
                    hypothesis_list.append(got)
            except tf.errors.OutOfRangeError:
                break
        outputs_tmp_filename = args.model_dir + \
            '{}.test.beam{}alpha{}.outputs.decodes'.format(\
            cur_mname, args.beam_width, args.alpha)
        refer_tmp_filename = args.model_dir + 'test_reference.tmp'
        with codecs.open(outputs_tmp_filename, 'w+', 'utf-8') as tmpfile, \
                codecs.open(refer_tmp_filename, 'w+', 'utf-8') as tmpreffile:
            for hyp, tgt in zip(hypothesis_list, targets_list):
                tmpfile.write(' '.join(hyp) + '\n')
                tmpreffile.write(' '.join(tgt) + '\n')
        test_bleu = float(100 * bleu_tool.bleu_wrapper(refer_tmp_filename, \
            outputs_tmp_filename, case_sensitive=True))
        test_loss = float(np.sum(np.array(test_loss)))
        print('test_bleu:%s test_loss:%s', \
            test_bleu, test_loss)
        logging.info('test_bleu:%s test_loss:%s', \
            test_bleu, test_loss)
        return {'loss': test_loss,
                'bleu': test_bleu}

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())
        var_list = tf.trainable_variables()
        with open(args.log_dir+'var.list', 'w+') as outfile:
            for var in var_list:
                outfile.write('var:{} shape:{} dtype:{}\n'.format(\
                    var.name, var.shape, var.dtype))
                logging.info('var:%s shape:%s', var.name, var.shape)
        writer = tf.summary.FileWriter(args.log_dir, graph=sess.graph)
        eval_writer = tf.summary.FileWriter(os.path.join(args.log_dir, \
            'eval/'), graph=sess.graph)
        lowest_loss, highest_bleu, best_epoch = -1, -1, -1
        if args.running_mode == 'train_and_evaluate':
            for epoch in range(args.max_train_epoch):
                if epoch % args.eval_interval_epoch != 0:
                    continue
                status = _train_epochs(sess, epoch, writer)
                eval_result = _eval_epoch(sess, epoch, eval_writer)
                eval_loss, eval_score = eval_result['loss'], eval_result['bleu']
                if args.eval_criteria == 'loss':
                    if lowest_loss < 0 or eval_loss < lowest_loss:
                        logging.info('the %s epoch got lowest loss %s', \
                            epoch, eval_loss)
                        eval_saver.save(sess, \
                            args.log_dir+'my-model-lowest_loss.ckpt')
                        lowest_loss = eval_loss
                        best_epoch = epoch
                elif args.eval_criteria == 'bleu':
                    if highest_bleu < 0 or eval_score > highest_bleu:
                        logging.info('the %s epoch, highest bleu %s', \
                            epoch, eval_score)
                        eval_saver.save(sess, \
                            args.log_dir + 'my-model-highest_bleu.ckpt')
                        highest_bleu = eval_score
                        best_epoch = epoch
                if status == 'finished':
                    logging.info('saving model for max training steps')
                    os.makedirs(args.log_dir+'/max/')
                    eval_saver.save(sess, \
                        args.log_dir+'/max/my-model-highest_bleu.ckpt')
                    break
        elif args.running_mode == 'test':
            if args.load_from_pytorch:
                modelpath = os.path.join(args.model_dir, args.model_filename)
                pytorch_params = pickle.load(open(modelpath, 'rb'))
                params = tf.trainable_variables()
                mname = modelpath.split('/')[-1]
                assert len(params) == len(pytorch_params)
                for param in params:
                    param_key = param.name
                    param_key = param_key.replace(':0', '')
                    sess.run(param.assign(pytorch_params[param_key]))
                print('loaded model from pytorch {}'.format(modelpath))
            elif args.model_dir == 'default':
                args.model_dir = args.log_dir
                logging.info('test model from:%s', args.model_dir)
                eval_saver.restore(sess,
                                   tf.train.latest_checkpoint(args.model_dir))
                mname =\
                    tf.train.latest_checkpoint(args.model_dir).split('/')[-1]
            elif args.model_dir == 'max':
                args.model_dir = args.log_dir + '/max/'
                logging.info('test model from:%s', args.model_dir)
                eval_saver.restore(sess, \
                    tf.train.latest_checkpoint(args.model_dir))
                mname = \
                    tf.train.latest_checkpoint(args.model_dir).split('/')[-1]
            else:
                logging.info('test model from %s', args.model_fullpath)
                mname = args.model_fullpaths.split('/')[-1]
                eval_saver.restore(sess, args.model_fullpath)
            logging.info('test data src:%s', args.test_src)
            logging.info('test data tgt:%s', args.test_tgt)
            _test_epoch(sess, mname)
        else:
            raise NotImplementedError
