"""
can run and give resule, but when exit:
    You must feed a value for placeholder tensor 'Placeholder' with dtype int32 and s
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name, no-name-in-module
import os
# We shall wrap all these modules
from texar.modules import TransformerEncoder, TransformerDecoder
from texar.losses import mle_losses
from texar.core import optimization as opt
# from texar import context
import numpy as np
import random
from data_load import load_test_data, load_de_vocab, load_en_vocab, data_hparams
import tensorflow as tf
import codecs
from nltk.translate.bleu_score import corpus_bleu
random.seed(123)
np.random.seed(123)
tf.set_random_seed(123)

if __name__ == "__main__":
    ### Build data pipeline

    # Config data hyperparams. Hyperparams not configured will be automatically
    # filled with default values. For text database, default values are defined
    # in `texar.data.database.default_text_dataset_hparams()`.
    extra_hparams = {
        'max_seq_length':10,
        'scale':True,
        'sinusoid':False,
        'embedding': {
            'initializer': {
                'type':'xavier_initializer',
                },
            'dim': 512,
        },
        'num_blocks': 6,
        'num_heads': 8,
    }
    X, Sources, Targets = load_test_data()
    x = tf.placeholder(tf.int32, shape=(data_hparams['batch_size'], data_hparams['max_seq_length']))
    y = tf.placeholder(tf.int32, shape=(data_hparams['batch_size'], data_hparams['max_seq_length']))
    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()

    encoder = TransformerEncoder(vocab_size=len(de2idx),
            hparams=extra_hparams)
    decoder = TransformerDecoder(vocab_size=len(idx2en),
            hparams=extra_hparams)


    print('encoder decoder finished')

    encoder_output = encoder(x)
    decoder_inputs = tf.concat((tf.ones_like(y[:,:1])*2, y[:,:-1]), -1)

    logits, preds = decoder(y, encoder_output)

    loss_params = {
            'label_smoothing':0.1,
    }

    is_target=tf.to_float(tf.not_equal(y, 0))
    y_smoothed = mle_losses.label_smoothing(y, len(idx2en), loss_params['label_smoothing'])
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_smoothed)

    mle_loss = tf.reduce_sum(loss*is_target)/(tf.reduce_sum(is_target))

    # Build train op. Only config the optimizer while using default settings
    # for other hyperparameters.
    opt_hparams={
        "optimizer": {
            "type": "AdamOptimizer",
            "kwargs": {
                "learning_rate": 0.0001,
                "beta1": 0.9,
                "beta2": 0.98,
                "epsilon": 1e-8,
            }
        }
    }
    train_op, global_step = opt.get_train_op(mle_loss, hparams=opt_hparams)

    ### Graph is done. Now start running
    sv = tf.train.Supervisor(graph=tf.get_default_graph(),
            logdir='logdir',
            save_model_secs=0)

    with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sv.saver.restore(sess, tf.train.latest_checkpoint('dutil_logdir'))
        print('restored!')

        mname = open('dutil_logdir/checkpoint', 'r').read().split('"')[1] # model name

        if not os.path.exists('dutil_results'): os.mkdir('dutil_results')
        with codecs.open("dutil_results/" + mname, "w", "utf-8") as fout:
            list_of_refs, hypotheses = [], []
            for i in range(len(X) // data_hparams['batch_size']):
                src = X[i*data_hparams['batch_size']: (i+1)*data_hparams['batch_size']]
                sources = Sources[i*data_hparams['batch_size']: (i+1)*data_hparams['batch_size']]
                targets = Targets[i*data_hparams['batch_size']: (i+1)*data_hparams['batch_size']]
                outputs = np.zeros((data_hparams['batch_size'], data_hparams['max_seq_length']), np.int32)
                for j in range(data_hparams['max_seq_length']):
                    print('expect to feed{} {} {} {}'.format(src.shape, src.dtype, outputs.shape, outputs.size))
                    _preds = sess.run(preds, feed_dict={x: src, y: outputs})
                    outputs[:, j] = _preds[:, j]

                for source, target, pred in zip(sources, targets, outputs): # sentence-wise
                    got = " ".join(idx2en[idx] for idx in pred).split("</S>")[0].strip()
                    fout.write("- source: " + source +"\n")
                    fout.write("- expected: " + target + "\n")
                    fout.write("- got: " + got + "\n\n")
                    fout.flush()

                # bleu score
                    ref = target.split()
                    hypothesis = got.split()
                    if len(ref) > 3 and len(hypothesis) > 3:
                        list_of_refs.append([ref])
                        hypotheses.append(hypothesis)

                ## Calculate bleu score
            score = corpus_bleu(list_of_refs, hypotheses)
            fout.write("Bleu Score = " + str(100*score))
        print('Done')
