"""
Example pipeline. This is a minimal example of basic RNN language model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name, no-name-in-module
import random
import copy
import pprint
import numpy as np
import tensorflow as tf
import logging
from texar.data import qPairedTextData
from texar.core.utils import _bucket_boundaries
from texar.modules import TransformerEncoder, TransformerDecoder
from texar.losses import mle_losses
#from texar.core import optimization as opt
from texar import context
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
    boundaries = _bucket_boundaries(max_length=256)
    #boundaries = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 26, 28, 30, 33, 36, 39, 42, 46, 50, 55, 60, 66, 72, 79, 86, 94, 103, 113, 124, 136, 149, 163, 179, 196, 215, 236]

    bucket_batch_size = [240, 180, 180, 180, 144, 144, 144, 120, 120, 120, 90, 90, 90, 90, 80, 72, 72, 60, 60, 48, 48, 48, 40, 40, 36, 30, 30, 24, 24, 20, 20, 18, 18, 16, 15, 12, 12, 10, 10, 9, 8, 8]
    data_hparams = {
        "num_epochs": 100,
        "seed": 123,
        "shuffle": True,
        "source_dataset": {
            "files": ['/home/hzt/shr/t2t_data/train_ende_wmt_bpe32k_en.txt.filtered'],
            "vocab_file": '/home/hzt/shr/t2t_data/vocab.bpe.32000.filtered',
            "processing": {
                "bos_token": "<BOS>",
                "eos_token": "<EOS>",
            }
        },
        "target_dataset": {
            "files": ['/home/hzt/shr/t2t_data/train_ende_wmt_bpe32k_de.txt.filtered'],
            "vocab_share":True,
        },
        'bucket_boundaries': boundaries,
        #'bucket_batch_size': [i // 4 for i  in bucket_batch_size],
        'bucket_batch_size': bucket_batch_size
    }
    hidden_dim = 512
    encoder_hparams = {
        'multiply_embedding_mode': "sqrt_depth",
        'embedding': {
            'name': 'lookup_table',
            'dim': hidden_dim,
            'initializer': {
                'type': 'uniform_unit_scaling',
            }
        },
        'max_seq_length':256,
        'sinusoid': True,
        #'sinusoid': False,
        'num_blocks': 6,
        'num_heads': 8,
        'poswise_feedforward': {
            'name':'ffn',
            'layers':[
                {
                    'type':'Conv1D',
                    'kwargs': {
                        'filters':hidden_dim*4,
                        'kernel_size':1,
                        'activation':'relu',
                        'use_bias':True,
                    }
                },
                {
                    'type':'Conv1D',
                    'kwargs': {
                        'filters':hidden_dim,
                        'kernel_size':1,
                        'use_bias':True,
                    }
                }
            ],
        },
    }
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
    decoder_hparams = copy.deepcopy(encoder_hparams)
    decoder_hparams['share_embed_and_transform'] = True
    decoder = TransformerDecoder(
        embedding = encoder._embedding,
        hparams=decoder_hparams)

    logits, preds = decoder(
        decoder_input,
        encoder_output,
        src_length=enc_input_length,
        tgt_length=dec_input_length)
    loss_params = {
        'label_smoothing':0.1,
    }
    smooth_labels = mle_losses.label_smoothing(labels, text_database.target_vocab.vocab_size, \
        loss_params['label_smoothing'])
    mle_loss = mle_losses.average_sequence_softmax_cross_entropy(
        labels=smooth_labels,
        logits=logits,
        sequence_length=labels_length)

    istarget = tf.to_float(tf.not_equal(labels, 0))
    acc = tf.reduce_sum(tf.to_float(tf.equal(tf.to_int64(preds), labels))*istarget) / tf.to_float((tf.reduce_sum(labels_length)))
    tf.summary.scalar('acc', acc)
    opt_hparams = {
        'learning_rate_schedule': 'linear_warmup_rsqrt_decay',
        #'learning_rate_schedule': 'static',
        'warmup_steps': 16000,
        'max_training_steps': 250000,
    }
    global_step = tf.Variable(0, trainable=False)

    fstep = tf.to_float(global_step)
    if opt_hparams['learning_rate_schedule'] == 'static':
        learning_rate = 1e-3
    else:
        learning_rate = 2 * tf.minimum(1.0, (fstep / opt_hparams['warmup_steps'])) \
            * tf.rsqrt(tf.maximum(fstep, opt_hparams['warmup_steps'])) \
            * encoder_hparams['embedding']['dim']**-0.5
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
            beta1=0.9, beta2=0.98, epsilon=1e-6)
    train_op = optimizer.minimize(mle_loss, global_step)
    tf.summary.scalar('lr', learning_rate)
    tf.summary.scalar('mle_loss', mle_loss)

    merged = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=5)
    config = tf.ConfigProto(
        allow_soft_placement=True)
    config.gpu_options.allow_growth=True
    vocab = text_database.source_vocab
    pp = pprint.PrettyPrinter(indent=4)

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
