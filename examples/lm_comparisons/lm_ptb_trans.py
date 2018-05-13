from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name, no-member, too-many-locals
import os
import importlib
import numpy as np
import tensorflow as tf
import texar as tx

from ptb_reader import prepare_data, ptb_iterator

flags = tf.flags

flags.DEFINE_string("data_path", "./",
                    "Directory containing PTB raw data (e.g., ptb.train.txt). "
                    "E.g., ./simple-examples/data. If not exists, "
                    "the directory will be created and PTB raw data will "
                    "be downloaded.")
flags.DEFINE_string("config", "config_small", "The config to use.")

FLAGS = flags.FLAGS

config = importlib.import_module(FLAGS.config)

def _main(_):
    # Data
    batch_size = config.batch_size
    num_steps = config.num_steps
    data = prepare_data(FLAGS.data_path)
    vocab_size = data["vocab_size"]

    inputs = tf.placeholder(tf.int32, [batch_size, num_steps])
    targets = tf.placeholder(tf.int32, [batch_size, num_steps])
    encoder_output = tf.placeholder(tf.int32, [batch_size, 1, config.emb.dim])

    embedder = tx.modules.WordEmbedder(
        vocab_size=vocab_size, hparams=config.emb)

    decoder = tx.modules.TransformerDecoder(
        embedding=embedder._embedding,
        hparams=config.decoder_hparams)

    logits, preds = decoder(
        decoder_input=inputs,
        encoder_output=encoder_output, #there should be 1-pos bias
        encoder_decoder_attention_bias=None,
    )
    #predictions = decoder.dynamic_decode(
    #    emb_inputs=inputs,
    #    encoder_decoder_attention_bias=None,
    #)

    # Losses & train ops
    mle_loss = tx.losses.sequence_sparse_softmax_cross_entropy(
        logits=logits,
        labels=targets,
        sequence_length=[num_steps] * batch_size)

    #l2_loss = sum([tf.nn.l2_loss(t) for t in tf.trainable_variables])
    # should we add the l2_loss on language model?

    global_step = tf.Variable(0, dtype=tf.int32)
    learning_rate = \
        tf.placeholder(dtype=tf.float32, shape=(), name='learning_rate')
    opt_vars = {
        'learning_rate': config.opt['init_lr'],
        'best_valid_ppl': 1e100,
        'steps_not_improved': 0,
    }
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        beta1=config.opt['Adam_beta1'],
        beta2=config.opt['Adam_beta2'],
        epsilon=config.opt['epsilon'])
    train_op = optimizer.minimize(mle_loss, global_step=global_step)
    def _run_epoch(sess, data_iter, is_train=False, verbose=True):
        loss = 0.
        iters = 0
        fetches = {
            "mle_loss": mle_loss,
            'global_step': global_step,
            'decoder_output': decoder.decoder_output,
        }
        if is_train:
            fetches["train_op"] = train_op

        mode = (tf.estimator.ModeKeys.TRAIN
                if is_train
                else tf.estimator.ModeKeys.EVAL)
        last_decoder_output = None

        for step, (x, y) in enumerate(data_iter):
            feed_dict = {
                inputs: x, targets: y,
                learning_rate: opt_vars['learning_rate'],
                tx.global_mode(): mode,
            }
            if step == 0:
                feed_dict[encoder_output] = np.ones((cur_batch_size,
                    1, config.embed.dim))
            else:
                feed_dict[encoder_output] = last_decoder_output[:, -1:, :]

            rets = sess.run(fetches, feed_dict)
            loss += rets["mle_loss"]
            iters += num_steps
            last_decoder_output = rets['decoder_output']
            ppl = np.exp(loss / iters)

            if is_train:
                print('global step:', rets['global_step'], ' '*4,
                      'training ppl:', ppl, file=training_log)
                training_log.flush()
            if is_train and rets['global_step'] % 100 ==0:
                valid_data_iter = ptb_iterator(
                    data['valid_text_id'], config.batch_size, num_steps)
                valid_ppl = _run_epoch(sess, valid_data_iter)
                test_data_iter = ptb_iterator(
                    data['test_text_id'], batch_size, num_steps)
                test_ppl = _run_epoch(sess, test_data_iter)
                print('global step:', rets['global_step'], ' '*4,
                      'learning_rate', opt_vars['learning_rate'], ' '*4,
                      'valid ppl:', valid_ppl, ' '*4,
                      'test ppl:', test_ppl,
                      file=eval_log)
                eval_log.flush()
                if valid_ppl < opt_vars['best_valid_ppl']:
                    opt_vars['best_valid_ppl'] = valid_ppl
                    opt_vars['steps_not_improved'] = 0
                else:
                    opt_vars['steps_not_improved'] += 1

                #TODO(haoran): this is a hard-coded hyperparameter so far.
                if opt_vars['steps_not_improved'] >= 30:
                    opt_vars['steps_not_improved'] = 0
                    opt_vars['learning_rate'] *= config.lr_decay

        ppl = np.exp(loss / iters)
        return ppl

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        for epoch in range(config.num_epochs):
            # Train
            train_data_iter = ptb_iterator(
                data["train_text_id"], config.batch_size, num_steps)
            train_ppl = _run_epoch(
                sess, train_data_iter, is_train=True, verbose=True)
            print("Epoch: %d Train Perplexity: %.3f" % (epoch, train_ppl))
            # Valid
            valid_data_iter = ptb_iterator(
                data["valid_text_id"], config.batch_size, num_steps)
            valid_ppl = _run_epoch(sess, valid_data_iter, epoch)
            print("Epoch: %d Valid Perplexity: %.3f" % (epoch, valid_ppl))
        # Test
        test_data_iter = ptb_iterator(
            data["test_text_id"], batch_size, num_steps)
        test_ppl = _run_epoch(sess, test_data_iter, 0)
        print("Test Perplexity: %.3f" % (test_ppl))

if __name__ == '__main__':
    LOG_DIR = 'language_model_trans_training_log/'
    os.system('mkdir ' + LOG_DIR)

    training_log = open(LOG_DIR + 'training_log.txt', 'w')
    eval_log = open(LOG_DIR + 'eval_log.txt', 'w')
    tf.app.run(main=_main)
