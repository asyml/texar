# Copyright 2018 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Example for building the Variational Autoencoder.

This is an impmentation of Variational Autoencoder for text generation

To run:

$ python vae_train.py

Hyperparameters and data path may be specified in config_trans.py

"""

# pylint: disable=invalid-name, no-member, too-many-locals
# pylint: disable=too-many-branches, too-many-statements, redefined-variable-type

import os
import sys
import time
import importlib
from io import open

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import texar.tf as tx


tfd = tfp.distributions

flags = tf.flags

flags.DEFINE_string("config", "config", "The config to use.")
flags.DEFINE_string("mode", "train", "train or predict")
flags.DEFINE_string("model", None, "model path for generating sentences")
flags.DEFINE_string("out", None, "generation output path")

FLAGS = flags.FLAGS

config = importlib.import_module(FLAGS.config)


def kl_dvg(means, logvars):
    """compute the KL divergence between Gaussian distribution
    """
    kl_cost = -0.5 * (logvars - tf.square(means) -
                      tf.exp(logvars) + 1.0)
    kl_cost = tf.reduce_mean(kl_cost, 0)

    return tf.reduce_sum(kl_cost)


def _main(_):
    # Data
    train_data = tx.data.MonoTextData(config.train_data_hparams)
    val_data = tx.data.MonoTextData(config.val_data_hparams)
    test_data = tx.data.MonoTextData(config.test_data_hparams)
    iterator = tx.data.TrainTestDataIterator(train=train_data,
                                             val=val_data,
                                             test=test_data)
    data_batch = iterator.get_next()

    opt_vars = {
        'learning_rate': config.lr_decay_hparams["init_lr"],
        'best_valid_nll': 1e100,
        'steps_not_improved': 0,
        'kl_weight': config.kl_anneal_hparams["start"]
    }

    decay_cnt = 0
    max_decay = config.lr_decay_hparams["max_decay"]
    decay_factor = config.lr_decay_hparams["decay_factor"]
    decay_ts = config.lr_decay_hparams["threshold"]

    save_dir = "./models/%s" % config.dataset

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    suffix = "%s_%sDecoder.ckpt" % \
            (config.dataset, config.decoder_type)

    save_path = os.path.join(save_dir, suffix)

    # KL term annealing rate
    anneal_r = 1.0 / (config.kl_anneal_hparams["warm_up"] *
                      (train_data.dataset_size() / config.batch_size))

    # Model architecture
    encoder_w_embedder = tx.modules.WordEmbedder(
        vocab_size=train_data.vocab.size, hparams=config.enc_emb_hparams)
    input_embed = encoder_w_embedder(data_batch["text_ids"])
    encoder = tx.modules.UnidirectionalRNNEncoder(
        hparams={"rnn_cell": config.enc_cell_hparams})

    decoder_w_embedder = tx.modules.WordEmbedder(
        vocab_size=train_data.vocab.size, hparams=config.dec_emb_hparams)
    output_w_embed = decoder_w_embedder(data_batch["text_ids"][:, :-1])

    if config.decoder_type == "lstm":
        output_embed = output_w_embed

        decoder = tx.modules.BasicRNNDecoder(
            vocab_size=train_data.vocab.size,
            hparams={"rnn_cell": config.dec_cell_hparams})
        decoder_initial_state_size = decoder.cell.state_size
    elif config.decoder_type == 'transformer':
        # position embedding
        decoder_p_embedder = tx.modules.SinusoidsPositionEmbedder(
            position_size=config.max_pos, hparams=config.dec_pos_emb_hparams)
        batch_size = tf.shape(data_batch["text_ids"])[0]
        max_seq_len = tf.shape(data_batch["text_ids"])[1] - 1
        batch_max_seq_len = tf.ones([batch_size], tf.int32) * max_seq_len
        output_p_embed = decoder_p_embedder(sequence_length=batch_max_seq_len)

        output_w_embed = output_w_embed * config.hidden_size ** 0.5
        output_embed = output_w_embed + output_p_embed

        # decoder
        decoder = tx.modules.TransformerDecoder(
            # tie word embedding with output layer
            output_layer=tf.transpose(decoder_w_embedder.embedding, (1, 0)),
            hparams=config.trans_hparams)
        decoder_initial_state_size = tf.TensorShape(
            [1, config.dec_emb_hparams["dim"]])
    else:
        raise NotImplementedError

    connector_mlp = tx.modules.MLPTransformConnector(
        config.latent_dims * 2)

    connector_stoch = tx.modules.ReparameterizedStochasticConnector(
        decoder_initial_state_size)

    # encoder -> connector -> decoder

    _, ecdr_states = encoder(
        input_embed,
        sequence_length=data_batch["length"])

    mean_logvar = connector_mlp(ecdr_states)
    mean, logvar = tf.split(mean_logvar, 2, 1)
    kl_loss = kl_dvg(mean, logvar)

    dst = tfd.MultivariateNormalDiag(
        loc=mean,
        scale_diag=tf.exp(0.5 * logvar))

    dcdr_states, latent_z = connector_stoch(dst)

    # decoder
    if config.decoder_type == "lstm":
        # concat latent variable to input at every time step
        latent_z = tf.expand_dims(latent_z, axis=1)
        latent_z = tf.tile(latent_z, [1, tf.shape(output_embed)[1], 1])
        output_embed = tf.concat([output_embed, latent_z], axis=2)

        outputs, _, _ = decoder(
            initial_state=dcdr_states,
            decoding_strategy="train_greedy",
            inputs=output_embed,
            sequence_length=data_batch["length"] - 1)
    else:
        outputs = decoder(
            inputs=output_embed,
            memory=dcdr_states,
            memory_sequence_length=tf.ones(tf.shape(dcdr_states)[0]))

    logits = outputs.logits

    seq_lengths = data_batch["length"] - 1
    # Losses & train ops
    rc_loss = tx.losses.sequence_sparse_softmax_cross_entropy(
        labels=data_batch["text_ids"][:, 1:],
        logits=logits,
        sequence_length=data_batch["length"] - 1)

    # KL annealing
    kl_weight = tf.placeholder(tf.float32, shape=())

    nll = rc_loss + kl_weight * kl_loss

    learning_rate = tf.placeholder(dtype=tf.float32, shape=(),
                                   name='learning_rate')
    train_op = tx.core.get_train_op(nll, learning_rate=learning_rate,
                                    hparams=config.opt_hparams)

    def _run_epoch(sess, epoch, mode_string, display=10):
        if mode_string == 'train':
            iterator.switch_to_train_data(sess)
        elif mode_string == 'valid':
            iterator.switch_to_val_data(sess)
        elif mode_string == 'test':
            iterator.switch_to_test_data(sess)

        step = 0
        start_time = time.time()
        num_words = num_sents = 0
        nll_ = 0.
        kl_loss_ = rc_loss_ = 0.

        while True:
            try:
                fetches = {"nll": nll,
                           "kl_loss": kl_loss,
                           "rc_loss": rc_loss,
                           "lengths": seq_lengths}

                if mode_string == 'train':
                    fetches["train_op"] = train_op
                    opt_vars["kl_weight"] = min(
                        1.0, opt_vars["kl_weight"] + anneal_r)

                    kl_weight_ = opt_vars["kl_weight"]
                else:
                    kl_weight_ = 1.0

                mode = (tf.estimator.ModeKeys.TRAIN if mode_string == 'train'
                        else tf.estimator.ModeKeys.EVAL)

                feed = {tx.global_mode(): mode,
                        kl_weight: kl_weight_,
                        learning_rate: opt_vars["learning_rate"]}

                fetches_ = sess.run(fetches, feed_dict=feed)

                batch_size_ = len(fetches_["lengths"])
                num_sents += batch_size_

                num_words += sum(fetches_["lengths"])
                nll_ += fetches_["nll"] * batch_size_
                kl_loss_ += fetches_["kl_loss"] * batch_size_
                rc_loss_ += fetches_["rc_loss"] * batch_size_

                if step % display == 0 and mode_string == 'train':
                    print('%s: epoch %d, step %d, nll %.4f, klw: %.4f, '
                          'KL %.4f,  rc %.4f, log_ppl %.4f, ppl %.4f, '
                          'time elapsed: %.1fs' %
                          (mode_string, epoch, step, nll_ / num_sents,
                           opt_vars["kl_weight"], kl_loss_ / num_sents,
                           rc_loss_ / num_sents, nll_ / num_words,
                           np.exp(nll_ / num_words), time.time() - start_time))

                    sys.stdout.flush()

                step += 1

            except tf.errors.OutOfRangeError:
                print('\n%s: epoch %d, nll %.4f, KL %.4f, rc %.4f, '
                      'log_ppl %.4f, ppl %.4f\n' %
                      (mode_string, epoch, nll_ / num_sents,
                       kl_loss_ / num_sents, rc_loss_ / num_sents,
                       nll_ / num_words, np.exp(nll_ / num_words)))
                break

        return nll_ / num_sents, np.exp(nll_ / num_words)

    def _generate(sess, saver, fname=None):
        if tf.train.checkpoint_exists(FLAGS.model):
            saver.restore(sess, FLAGS.model)
        else:
            raise ValueError("cannot find checkpoint model")

        batch_size = train_data.batch_size

        dst = tfd.MultivariateNormalDiag(
            loc=tf.zeros([batch_size, config.latent_dims]),
            scale_diag=tf.ones([batch_size, config.latent_dims]))

        dcdr_states, latent_z = connector_stoch(dst)

        vocab = train_data.vocab
        start_tokens = tf.ones(batch_size, tf.int32) * vocab.bos_token_id
        end_token = vocab.eos_token_id

        if config.decoder_type == "lstm":
            def _cat_embedder(ids):
                """Concatenates latent variable to input word embeddings
                """
                embedding = decoder_w_embedder(ids)
                return tf.concat([embedding, latent_z], axis=1)

            outputs, _, _ = decoder(
                initial_state=dcdr_states,
                decoding_strategy="infer_sample",
                embedding=_cat_embedder,
                max_decoding_length=100,
                start_tokens=start_tokens,
                end_token=end_token)
        else:
            def _embedding_fn(ids, times):
                w_embed = decoder_w_embedder(ids)
                p_embed = decoder_p_embedder(times)
                return w_embed * config.hidden_size ** 0.5 + p_embed

            outputs, _ = decoder(
                memory=dcdr_states,
                decoding_strategy="infer_sample",
                memory_sequence_length=tf.ones(tf.shape(dcdr_states)[0]),
                embedding=_embedding_fn,
                max_decoding_length=100,
                start_tokens=start_tokens,
                end_token=end_token)

        sample_tokens = vocab.map_ids_to_tokens(outputs.sample_id)
        sess.run(tf.tables_initializer())

        feed = {tx.global_mode(): tf.estimator.ModeKeys.PREDICT}
        sample_tokens_ = sess.run(sample_tokens, feed_dict=feed)

        if fname is None:
            fh = sys.stdout
        else:
            fh = open(fname, 'w', encoding='utf-8')

        for sent in sample_tokens_:
            sent = tx.utils.compat_as_text(list(sent))
            end_id = len(sent)
            if vocab.eos_token in sent:
                end_id = sent.index(vocab.eos_token)
            fh.write(' '.join(sent[:end_id + 1]) + '\n')

        print('Output done')
        fh.close()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        # generate samples from prior
        if FLAGS.mode == "predict":
            _generate(sess, saver, FLAGS.out)
            return

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        # Counts trainable parameters
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()  # shape is an array of tf.Dimension
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print("%d total parameters" % total_parameters)

        best_nll = best_ppl = 0.

        for epoch in range(config.num_epochs):
            _, _ = _run_epoch(sess, epoch, 'train', display=200)
            val_nll, _ = _run_epoch(sess, epoch, 'valid')
            test_nll, test_ppl = _run_epoch(sess, epoch, 'test')

            if val_nll < opt_vars['best_valid_nll']:
                opt_vars['best_valid_nll'] = val_nll
                opt_vars['steps_not_improved'] = 0
                best_nll = test_nll
                best_ppl = test_ppl
                saver.save(sess, save_path)
            else:
                opt_vars['steps_not_improved'] += 1
                if opt_vars['steps_not_improved'] == decay_ts:
                    old_lr = opt_vars['learning_rate']
                    opt_vars['learning_rate'] *= decay_factor
                    opt_vars['steps_not_improved'] = 0
                    new_lr = opt_vars['learning_rate']

                    print('-----\nchange lr, old lr: %f, new lr: %f\n-----' %
                          (old_lr, new_lr))

                    saver.restore(sess, save_path)

                    decay_cnt += 1
                    if decay_cnt == max_decay:
                        break

        print('\nbest testing nll: %.4f, best testing ppl %.4f\n' %
              (best_nll, best_ppl))


if __name__ == '__main__':
    tf.app.run(main=_main)
