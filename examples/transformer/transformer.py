"""
Example pipeline. This is a minimal example of basic RNN language model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name, no-name-in-module
import random
import copy
import numpy as np
import tensorflow as tf
import logging
from texar.data import qPairedTextData
from texar.core.utils import _bucket_boundaries
from texar.modules import TransformerEncoder, TransformerDecoder
from texar.losses import mle_losses
#from texar.core import optimization as opt
from texar import context
from hyperparams import dataset_hparams, encoder_hparams, decoder_hparams, \
    opt_hparams, loss_hparams
def config_logging(filepath):
    logging.basicConfig(filename = filepath+'logging.txt', \
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',\
        datefmt='%a, %d %b %Y %H:%M:%S',\
        level=logging.INFO)

if __name__ == "__main__":
    ### Build data pipeline
    logdir = './logdir/'
    config_logging(logdir)

    tf.set_random_seed(123)
    np.random.seed(123)
    random.seed(123)
    hidden_dim = 512
    # Construct the database
    text_database = qPairedTextData(data_hparams)
    text_data_batch = text_database()
    ori_src_text = text_data_batch['source_text_ids']
    ori_tgt_text = text_data_batch['target_text_ids']

    encoder_input = ori_src_text[:, 1:]
    decoder_input = ori_tgt_text[:, :-1]
    labels = ori_tgt_text[:, 1:]

    enc_input_length = tf.reduce_sum(tf.to_float(tf.not_equal(encoder_input, 0)), axis=-1)
    dec_input_length = tf.reduce_sum(tf.to_float(tf.not_equal(decoder_input, 0)), axis=-1)
    labels_length = tf.reduce_sum(tf.to_float(tf.not_equal(labels, 0)), axis=-1)

    enc_input_length = tf.Print(enc_input_length,
        data=[tf.shape(ori_src_text), tf.shape(ori_tgt_text), enc_input_length, dec_input_length, labels_length])

    encoder = TransformerEncoder(
        vocab_size=text_database.source_vocab.vocab_size,\
        hparams=encoder_hparams)
    encoder_output = encoder(encoder_input, inputs_length=enc_input_length)
    decoder = TransformerDecoder(
        embedding = encoder._embedding,
        hparams=decoder_hparams)

    logits, preds = decoder(
        decoder_input,
        encoder_output,
        src_length=enc_input_length,
        tgt_length=dec_input_length)
    smooth_labels = mle_losses.label_smoothing(labels, text_database.target_vocab.vocab_size, \
        loss_hparams['label_smoothing'])
    mle_loss = mle_losses.average_sequence_softmax_cross_entropy(
        labels=smooth_labels,
        logits=logits,
        sequence_length=labels_length)

    istarget = tf.to_float(tf.not_equal(labels, 0))
    acc = tf.reduce_sum(tf.to_float(tf.equal(tf.to_int64(preds), labels))*istarget) / tf.to_float((tf.reduce_sum(labels_length)))
    tf.summary.scalar('acc', acc)
    global_step = tf.Variable(0, trainable=False)

    fstep = tf.to_float(global_step)
    if opt_hparams['learning_rate_schedule'] == 'static':
        learning_rate = 1e-3
    else:
        learning_rate = 2 * tf.minimum(1.0, (fstep / opt_hparams['warmup_steps'])) \
            * tf.rsqrt(tf.maximum(fstep, opt_hparams['warmup_steps'])) \
            * encoder_hparams['embedding']['dim']**-0.5
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
            beta1=0.9, beta2=0.997, epsilon=1e-9)
    train_op = optimizer.minimize(mle_loss, global_step)
    tf.summary.scalar('lr', learning_rate)
    tf.summary.scalar('mle_loss', mle_loss)

    merged = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=5)
    config = tf.ConfigProto(
        allow_soft_placement=True)
    config.gpu_options.allow_growth=True
    vocab = text_database.source_vocab

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())
        graph = tf.get_default_graph()
        graph.finalize()

        var_list = tf.trainable_variables()
        with open(logdir+'var.list', 'w+') as outfile:
            for var in var_list:
                outfile.write('var:{} shape:{} dtype:{}\n'.format(var.name, var.shape, var.dtype))
                logging.info('var:{} shape:{}'.format(var.name, var.shape, var.dtype))
        writer = tf.summary.FileWriter(logdir, graph=sess.graph)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            while not coord.should_stop():
                source, dec_in, target, predict, _, step, loss, mgd = sess.run(
                    [encoder_input, decoder_input, labels, preds, train_op, global_step, mle_loss, merged],
                    feed_dict={context.is_train(): True})
                if step % 100 == 0:
                    logging.info('step:{} source:{} targets:{} loss:{}'.format(\
                        step, source.shape, target.shape, loss))
                source, dec_in, target = source.tolist(), dec_in.tolist(), target.tolist()
                swords = [ ' '.join([vocab._id_to_token_map_py[i] for i in sent]) for sent in source ]
                dwords = [ ' '.join([vocab._id_to_token_map_py[i] for i in sent]) for sent in dec_in ]
                twords = [ ' '.join([vocab._id_to_token_map_py[i] for i in sent]) for sent in target ]
                writer.add_summary(mgd, global_step=step)
                if step % 1000 == 0:
                    print('step:{} loss:{}'.format(step, loss))
                    saver.save(sess, logdir+'my-model', global_step=step)
                if step == opt_hparams['max_training_steps']:
                    coord.request_stop()
        except tf.errors.OutOfRangeError:
            print('Done -- epoch limit reached')
        finally:
            coord.request_stop()
        coord.join(threads)
        saver.save(sess, logdir+'my-model', global_step=step)
