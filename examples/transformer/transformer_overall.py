"""
Example pipeline. This is a minimal example of basic RNN language model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name, no-name-in-module
import random
import numpy as np
import tensorflow as tf
import logging
import texar as tx
import codecs
from texar.modules import TransformerEncoder, TransformerDecoder
from texar.losses import mle_losses
#from texar.core import optimization as opt
from texar import context
from hyperparams import train_dataset_hparams, test_dataset_hparams, encoder_hparams, decoder_hparams, \
    opt_hparams, loss_hparams, args
def config_logging(filepath):
    logging.basicConfig(filename = filepath+'logging.txt', \
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',\
        datefmt='%a, %d %b %Y %H:%M:%S',\
        level=logging.INFO)

if __name__ == "__main__":
    ### Build data pipeline
    config_logging(args.log_dir)
    tf.set_random_seed(1234)
    np.random.seed(1234)
    random.seed(1234)
    hidden_dim = 512
    # Construct the database
    train_database = tx.data.PairedTextData(train_dataset_hparams)
    #eval_database = qPairedTextData(eval_dataset_hparams)
    test_database  =tx.data.PairedTextData(test_dataset_hparams)
    iterator = tx.data.TrainTestDataIterator(train=train_database,
                      test=test_database)
    text_data_batch = iterator.get_next()

    ori_src_text = text_data_batch['source_text_ids']
    ori_tgt_text = text_data_batch['target_text_ids']

    encoder_input = ori_src_text[:, 1:]
    decoder_input = ori_tgt_text[:, :-1]
    labels = ori_tgt_text[:, 1:]

    enc_input_length = tf.reduce_sum(tf.to_float(tf.not_equal(encoder_input, 0)), axis=-1)
    dec_input_length = tf.reduce_sum(tf.to_float(tf.not_equal(decoder_input, 0)), axis=-1)
    #enc_input_length = tf.Print(enc_input_length,
    #    data=[tf.shape(ori_src_text), tf.shape(ori_tgt_text), enc_input_length, dec_input_length, labels_length])

    encoder = TransformerEncoder(
        vocab_size=train_database.source_vocab.size,\
        hparams=encoder_hparams)
    encoder_output, encoder_decoder_attention_bias = encoder(encoder_input, inputs_length=enc_input_length)
    decoder = TransformerDecoder(
        embedding = encoder._embedding,
        hparams=decoder_hparams)

    logits, preds = decoder(
        decoder_input,
        encoder_output,
        encoder_decoder_attention_bias,
    )
    predictions = decoder.dynamic_decode(
        encoder_output,
        encoder_decoder_attention_bias,
    )

    mle_loss = mle_losses.smoothing_cross_entropy(logits, labels, train_database.target_vocab.size,
        loss_hparams['label_confidence'])
    istarget = tf.to_float(tf.not_equal(labels, 0))
    mle_loss = tf.reduce_sum(mle_loss * istarget) / tf.reduce_sum(istarget)

    acc = tf.reduce_sum(tf.to_float(tf.equal(tf.to_int64(preds), labels))*istarget) / tf.to_float((tf.reduce_sum(istarget)))
    tf.summary.scalar('acc', acc)
    global_step = tf.Variable(0, trainable=False)

    fstep = tf.to_float(global_step)
    if opt_hparams['learning_rate_schedule'] == 'static':
        learning_rate = 1e-3
    else:
        learning_rate = opt_hparams['lr_constant'] \
            * tf.minimum(1.0, (fstep / opt_hparams['warmup_steps'])) \
            * tf.rsqrt(tf.maximum(fstep, opt_hparams['warmup_steps'])) \
            * encoder_hparams['embedding']['dim']**-0.5
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        beta1=opt_hparams['Adam_beta1'],
        beta2=opt_hparams['Adam_beta2'],
        epsilon=opt_hparams['Adam_epsilon'],
    )
    train_op = optimizer.minimize(mle_loss, global_step)
    tf.summary.scalar('lr', learning_rate)
    tf.summary.scalar('mle_loss', mle_loss)

    merged = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=5)
    config = tf.ConfigProto(
        allow_soft_placement=True)
    config.gpu_options.allow_growth=True
    vocab = train_database.source_vocab


    graph = tf.get_default_graph()
    graph.finalize()

    var_list = tf.trainable_variables()
    #with open(args.log_dir+'var.list', 'w+') as outfile:
    #    for var in var_list:
    #        outfile.write('var:{} shape:{} dtype:{}\n'.format(var.name, var.shape, var.dtype))
    #            logging.info('var:{} shape:{}'.format(var.name, var.shape, var.dtype))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())
        writer = tf.summary.FileWriter(args.log_dir, graph=sess.graph)
        if args.running_mode == 'train':
            for epoch in range(args.max_train_epoch):
                _train_epochs(sess, epoch)
                _test_epoch(sess, epoch)
        elif args.running_mode == 'test':
            _test_epoch(sess, epoch)
    def _test_epoch(sess, epoch):
        iterator.switch_to_test_data(sess)
        sources_list, targets_list, hypothesis_list = [], [], []
        while True:
            try:
                fetches = {
                    'predictions': predictions,
                    'source': ori_src_text,
                    'target': ori_tgt_text,
                    'step': global_step,
                }
                feed = {context.global_mode(): tf.estimator.ModeKeys.PREDICT}
                _fetches = sess.run(fetches, feed_dict=feed)
                sources, sampled_ids, targets = \
                    _fetches['source'].tolist(), \
                    _fetches['predictions']['sampled_ids'][:, 0, :].tolist(), \
                    _fetches['target'][:, 1:].tolist()
                def _id2word_map(id_arrays):
                    return [' '.join([vocab._id_to_token_map_py[i] for i in sent]) for sent in id_arrays]
                sources, targets, dwords = \
                    _id2word_map(sources), _id2word_map(targets), _id2word_map(sampled_ids)
                for source, target, pred in zip(sources, targets, dwords):
                    source = source.split('<EOS>')[0].strip().split()
                    target = target.split('<EOS>')[0].strip().split()
                    got = pred.split('<EOS>')[0].strip().split()
                    sources_list.append(source)
                    targets_list.append(target)
                    hypothesis_list.append(got)
            except tf.errors.OutOfRangeError:
                break
        if args.save_eval_output:
            with codecs.open(args.model_path + 'my_model_epoch{}.beam{}alpha{}.outputs', 'w+', 'utf-8') as outputfile:
                for sent in hypothesis_list:
                    outputfile.write(' '.join(sent) + '\n')

    def _train_epochs(sess, epoch, writer):
        iterator.switch_to_train_data(sess)
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
                _fetches = sess.run(fetches, feed_dict=feed)
                step, source, target, loss, mgd = _fetches['step'], _fetches['source'], \
                    _fetches['target'], _fetches['loss'], _fetches['mgd']
                if step % 100 == 0:
                    logging.info('step:{} source:{} targets:{} loss:{}'.format(\
                        step, source.shape, target.shape, loss))
                writer.add_summary(mgd, global_step=step)
                if step % 1000 == 0:
                    print('step:{} loss:{}'.format(step, loss))
                    saver.save(sess, args.log_dir+'my-model', global_step=step)
            except tf.errors.OutOfRangeError:
                break
        saver.save(sess, args.log_dir+'my-model-epoch{}'.format(epoch), global_step=step)
