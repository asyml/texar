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
from texar.data import PairedTextDataBase
from texar.core.utils import _bucket_boundaries
from texar.modules import TransformerEncoder, TransformerDecoder
from texar.losses import mle_losses
#from texar.core import optimization as opt
from texar import context

if __name__ == "__main__":
    ### Build data pipeline
    tf.set_random_seed(123)
    np.random.seed(123)
    random.seed(123)
    boundaries = _bucket_boundaries(max_length=256)
    bucket_batch_size = [240, 180, 180, 180, 144, 144, 144, 120, 120, 120, 90, 90, 90, 90, 80, 72, 72, 60, 60, 48, 48, 48, 40, 40, 36, 30, 30, 24, 24, 20, 20, 18, 18, 16, 15, 12, 12, 10, 10, 9, 8, 8]
    data_hparams = {
        "num_epochs": 2,
        "seed": 123,
        "shuffle": True,
        "source_dataset": {
            "files": ['/home/shr/t2t_data/train_ende_wmt_bpe32k_en.txt'],
            "vocab_file": '/home/shr/t2t_data/vocab.bpe.32000.filtered',
            "processing": {
                "bos_token": "<BOS>",
                "eos_token": "<EOS>",
            }
        },
        "target_dataset": {
            "files": ['/home/shr/t2t_data/train_ende_wmt_bpe32k_de.txt'],
            "vocab_share":True,
            "processing":{
                "bos_token": "<BOS>",
                "eos_token": "<EOS>",
            },
        },
        #'batch_size':50,
        #'batch_size':32,
        'bucket_boundaries': [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 26, 28, 30, 33, 36, 39, 42, 46, 50, 55, 60, 66, 72, 79, 86, 94, 103, 113, 124, 136, 149, 163, 179, 196, 215, 236],
        'bucket_batch_size':[i // 2 for i in bucket_batch_size]
        #'bucket_batch_size': bucket_batch_size,
    }
    hidden_dim =512
    extra_hparams = {
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
    text_database = PairedTextDataBase(data_hparams)
    text_data_batch = text_database()
    ori_src_text = text_data_batch['source_text_ids']
    ori_tgt_text = text_data_batch['target_text_ids']

    #ori_src_text = text_data_batch['source_text_ids'][:, :extra_hparams['max_seq_length']]
    #ori_tgt_text = text_data_batch['target_text_ids'][:, :extra_hparams['max_seq_length']]

    encoder_input = ori_src_text[:, 1:]
    decoder_input = ori_tgt_text[:, :-1]

    src_length = tf.reduce_sum(tf.to_float(tf.not_equal(encoder_input, 0)), axis=-1)
    tgt_length = tf.reduce_sum(tf.to_float(tf.not_equal(decoder_input, 0)), axis=-1)

    encoder = TransformerEncoder(vocab_size=text_database.source_vocab.vocab_size,\
        hparams=extra_hparams)
    encoder_output = encoder(encoder_input, inputs_length=src_length)

    decoder = TransformerDecoder(
        vocab_size=text_database.target_vocab.vocab_size,\
        embedding = encoder._embedding,
        hparams=extra_hparams)
    logits, preds = decoder(
        decoder_input,
        encoder_output,
        src_length=src_length,
        tgt_length=tgt_length)
    print('logits:{} preds:{}'.format(logits.shape, preds.shape))
    loss_params = {
        'label_smoothing':0.1,
    }
    labels = ori_tgt_text[:, 1:]
    smooth_labels = mle_losses.label_smoothing(labels, text_database.target_vocab.vocab_size, \
        loss_params['label_smoothing'])
    mle_loss = mle_losses.average_sequence_softmax_cross_entropy(
        labels=smooth_labels,
        logits=logits,
        sequence_length=tgt_length)

    opt_hparams = {
        'warmup_steps': 16000,
    }
    global_step = tf.Variable(0, trainable=False)
    fstep = tf.to_float(global_step)
    learning_rate = extra_hparams['embedding']['dim']**-0.5 *\
        (tf.minimum((fstep+1)**-0.5,
            (fstep+1) * opt_hparams['warmup_steps']**-1.5) )
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
            beta1=0.9, beta2=0.98, epsilon=1e-8)
    train_op = optimizer.minimize(mle_loss, global_step)
    tf.summary.scalar('lr', learning_rate)
    tf.summary.scalar('mle_loss', mle_loss)

    merged = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=2)
    with tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())
        graph = tf.get_default_graph()
        graph.finalize()
        var_list = tf.trainable_variables()
        #for var in var_list:
        #    print('var:{} shape:{} dtype:{}'.format(var.name, var.shape, var.dtype))
        #exit()
        writer = tf.summary.FileWriter("./logdir/", graph=sess.graph)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                #source, target= sess.run(
                #    [encoder_input, labels], feed_dict={context.is_train():True})
                source, target, predict, _, step, loss, mgd = sess.run(
                    [encoder_input, labels, preds, train_op, global_step, mle_loss, merged],
                    feed_dict={context.is_train(): True})
                writer.add_summary(mgd, global_step=step)
                print('step:{} source:{} target:{}'.format(step, source.shape, target.shape))
                if step % 100 == 0:
                    print('step:{} loss:{}'.format(step, loss))
                    saver.save(sess, './logdir/my-model', global_step=step)
        except tf.errors.OutOfRangeError:
            print('Done -- epoch limit reached')
        finally:
            coord.request_stop()
        coord.join(threads)
        #saver.save(sess, './logdir/my-model', global_step=step)
