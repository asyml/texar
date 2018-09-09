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
"""SeqGAN for language modeling
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name, no-member, too-many-locals

import importlib
import numpy as np
import tensorflow as tf
import texar as tx

flags = tf.flags
flags.DEFINE_string("dataset", "ptb",
                    "perform training on ptb or coco.")
flags.DEFINE_string("data_path", "./",
                    "Directory containing coco. If not exists, "
                    "the directory will be created, and the data "
                    "will be downloaded.")
flags.DEFINE_string("config", "config_ptb_small", "The config to use.")
FLAGS = flags.FLAGS

config = importlib.import_module(FLAGS.config)


def _main(_):
    log = open(config.log_file, 'w')
    bleu_log = open(config.bleu_file, 'w')

    # Data
    train_data = tx.data.MonoTextData(config.train_data_hparams)
    val_data = tx.data.MonoTextData(config.val_data_hparams)
    test_data = tx.data.MonoTextData(config.test_data_hparams)
    iterator = tx.data.TrainTestDataIterator(train=train_data,
                                             val=val_data,
                                             test=test_data)
    data_batch = iterator.get_next()

    batch_size = tf.shape(data_batch["text_ids"])[0]
    num_steps = tf.shape(data_batch["text_ids"])[1]
    vocab_size = train_data.vocab.size

    # Model architecture
    g_embedder = tx.modules.WordEmbedder(vocab_size=vocab_size,
                                         hparams=config.emb_hparams)
    input_embed = g_embedder(data_batch["text_ids"][:, :-1])

    if config.enc_keep_prob_in < 1:
        input_embed = tf.nn.dropout(
            input_embed, tx.utils.switch_dropout(config.enc_keep_prob_in))

    decoder = tx.modules.BasicRNNDecoder(
        vocab_size=vocab_size,
        hparams={"rnn_cell": config.dec_cell_hparams,
                 "max_decoding_length_infer": config.max_num_steps + 2})
    initial_state = decoder.zero_state(batch_size=batch_size,
                                       dtype=tf.float32)

    # ------------Pretrain Generator---------------
    outputs, _, _ = decoder(
        initial_state=initial_state,
        decoding_strategy="train_greedy",
        inputs=input_embed,
        sequence_length=data_batch["length"] - 1)

    mle_loss = tx.losses.sequence_sparse_softmax_cross_entropy(
        labels=data_batch["text_ids"][:, 1:],
        logits=outputs.logits,
        sequence_length=data_batch["length"] - 1)

    global_step = tf.Variable(0, trainable=False)
    gen_train_op = tx.core.get_train_op(mle_loss,
                                        global_step=global_step,
                                        increment_global_step=True,
                                        hparams=config.g_opt_hparams)

    # -------------Generator Infer-------------------
    start_tokens = tf.cast(tf.fill([batch_size],
                                   train_data.vocab.bos_token_id),
                           dtype=tf.int32)
    infer_outputs, _, sequence_length = decoder(
        decoding_strategy="infer_sample",
        start_tokens=start_tokens,
        end_token=train_data.vocab.eos_token_id,
        embedding=g_embedder,
        initial_state=initial_state,
        max_decoding_length=config.max_num_steps)

    infer_logits = infer_outputs.logits
    infer_sample_ids = infer_outputs.sample_id

    # ------------Pretrain Discriminator---------------
    discriminator = tx.modules.UnidirectionalRNNClassifier(
        hparams={"clas_strategy": "time_wise", "num_classes": 1})
    d_embedder = tx.modules.WordEmbedder(vocab_size=vocab_size,
                                         hparams=config.emb_hparams)

    r_logits, _ = discriminator(d_embedder(data_batch["text_ids"]))
    f_logits, _ = discriminator(d_embedder(infer_sample_ids))

    r_loss = tx.losses.sequence_sigmoid_cross_entropy(
        labels=tf.ones_like(data_batch["text_ids"], dtype=tf.float32),
        logits=tf.squeeze(r_logits),
        sequence_length=data_batch["length"])  # r_preds -> 1.
    f_loss = tx.losses.sequence_sigmoid_cross_entropy(
        labels=tf.zeros_like(infer_sample_ids, dtype=tf.float32),
        logits=tf.squeeze(f_logits),
        sequence_length=sequence_length)  # infer_logits -> 0.
    dis_loss = r_loss + f_loss
    dis_loss.set_shape(())

    dis_train_op = tx.core.get_train_op(dis_loss,
                                        global_step=global_step,
                                        increment_global_step=False,
                                        hparams=config.d_opt_hparams)

    # ------------Adeversarial---------------
    infer_logits = tf.clip_by_value(
        tf.nn.softmax(infer_logits) *
        tf.one_hot(infer_sample_ids, vocab_size), 1e-20, 1)

    expected_reward = tf.Variable(tf.zeros((config.max_num_steps,)))
    reward = tf.reshape(f_logits, shape=(batch_size, -1)) - \
            expected_reward[:tf.shape(f_logits)[1]]
    mean_reward = tf.reduce_mean(reward)
    exp_reward_loss = -tf.reduce_mean(tf.abs(reward))
    exp_reward_loss.set_shape(())
    exp_op = tx.core.get_train_op(exp_reward_loss,
                                  global_step=global_step,
                                  increment_global_step=False,
                                  hparams=config.update_opt_hparams)
    reward = tx.losses.discount_reward(
        reward, sequence_length=tf.squeeze(sequence_length), tensor_rank=2)
    update_loss = tf.reduce_mean(tf.log(infer_logits) *
                                 tf.expand_dims(reward, -1))
    update_loss.set_shape(())
    gen_op = tx.core.get_train_op(update_loss,
                                  global_step=global_step,
                                  increment_global_step=True,
                                  hparams=config.update_opt_hparams)
    update_op = tf.group(gen_op, exp_op)

    def _g_train_epoch(sess, epoch, mode_string):
        iterator.switch_to_train_data(sess)
        while True:
            try:
                if mode_string == 'train':
                    fetches = {
                        'mean_rwd': mean_reward,
                        'exp_rwd_loss': exp_reward_loss,
                        'update_loss': update_loss,
                        'update_op': update_op,
                        'exp_rwd': expected_reward,
                        'step': global_step
                    }
                elif mode_string == 'pretrain':
                    fetches = {
                        'mle_loss': mle_loss,
                        'num_steps': num_steps,
                        'train_op': gen_train_op,
                        'step': global_step
                    }
                else:
                    raise ValueError(
                        "Expect mode_string to be one of "
                        "['pretrain', 'train'], got %s" % mode_string)
                rtns = sess.run(fetches)
                step = rtns['step']
                if step % 200 == 1:
                    if mode_string == 'pretrain':
                        ppl = np.exp(rtns['mle_loss'] / rtns["num_steps"])
                        rst = "G {0:6s} epoch {1:3d}, step {2:3d}:" \
                              " train_ppl: {3:6f}".format(mode_string,
                                                          epoch, step, ppl)
                    else:
                        rst = "G {0:6s} epoch {1:3d}, step {2:3d}: " \
                              "mean_reward: {3:6f}, " \
                              "expect_reward_loss:{4:6f}, " \
                              "update_loss: {5:6f}".format(
                                  mode_string, epoch, step, rtns['mean_rwd'],
                                  rtns['exp_rwd_loss'], rtns['update_loss'])
                    log.write(rst + '\n')
                    log.flush()
                    print(rst)
                    if mode_string == 'train':  # a batch per adversarial epoch
                        break
            except tf.errors.OutOfRangeError:
                break
        return

    def _g_test_epoch(sess, epoch, mode_string):
        def _id2word_map(id_arrays):
            return [' '.join([train_data.vocab.id_to_token_map_py[i]
                              for i in sent]) for sent in id_arrays]

        if mode_string == 'valid':
            iterator.switch_to_val_data(sess)
        elif mode_string == 'test':
            iterator.switch_to_test_data(sess)
        else:
            raise ValueError("Expect mode_string to be one of "
                             "['valid', 'test'], got %s" % mode_string)

        target_list, inference_list = [], []
        loss, steps = 0., 0
        while True:
            try:
                fetches = {
                    "mle_loss": mle_loss,
                    "num_steps": num_steps
                }
                if mode_string == 'test':
                    fetches['target_sample_id'] = data_batch["text_ids"]
                    fetches['infer_sample_id'] = infer_sample_ids

                feed_dict = {tx.global_mode(): tf.estimator.ModeKeys.EVAL}

                rtns = sess.run(fetches, feed_dict)

                loss += rtns['mle_loss']
                steps += rtns['num_steps']

                if mode_string == 'test':
                    targets = _id2word_map(rtns['target_sample_id'].tolist())
                    for t in targets:
                        target_list.extend(t.split('<EOS>')[0].strip().split())

                    inferences = _id2word_map(rtns['infer_sample_id'].tolist())
                    for inf in inferences:
                        inference_list.extend( # remove <BOS>
                            inf.split('<EOS>')[0].strip().split()[1:])

            except tf.errors.OutOfRangeError:
                break

        ppl = np.exp(loss / steps)
        rst = "G {0:6s} epoch {1:3d}, step {2:3s}:" \
              " {3:5s}_ppl: {4:6f}"\
            .format(mode_string, epoch, '-', mode_string, ppl)
        log.write(rst + '\n')
        log.flush()
        print(rst)

        if mode_string == 'test':
            bleu_test = tx.evals.sentence_bleu_moses(
                references=[target_list],
                hypothesis=inference_list,
                lowercase=True, return_all=True)
            if not isinstance(bleu_test, np.ndarray):  # might return 0.0 if inference_list is null
                bleu_test = [bleu_test] * 5
            rst_test = "epoch %d BLEU1~4 on test dataset:\n" \
                       "%f\n%f\n%f\n%f\n\n" % \
                       (epoch, bleu_test[1], bleu_test[2],
                        bleu_test[3], bleu_test[4])
            print(rst_test)
            bleu_log.write(rst_test)
            bleu_log.flush()

        return

    def _d_run_epoch(sess, epoch, mode_string='pretrain'):
        iterator.switch_to_train_data(sess)
        step = 0
        while True:
            try:
                fetches = {
                    "mle_loss": dis_loss,
                    "r_loss": r_loss,
                    "f_loss": f_loss,
                    "train_op": dis_train_op
                }
                rtns = sess.run(fetches)
                if step % 200 == 0:
                    rst = "D {0:6s} epoch {1:3d}, step {2:3d}: " \
                          "dis_total_loss: {3:6f}, r_loss: {4:6f}, " \
                          "f_loss: {5:6f}".format(
                              mode_string, epoch, step, rtns['mle_loss'],
                              rtns['r_loss'], rtns['f_loss'])
                    log.write(rst + '\n')
                    log.flush()
                    print(rst)
                step += 1
                if step == 15 and mode_string == 'train':
                    break
            except tf.errors.OutOfRangeError:
                break

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        # Generator pre-training
        for g_epoch in range(config.generator_pretrain_epoch):
            _g_train_epoch(sess, g_epoch, 'pretrain')
            if g_epoch % 10 == 0 or \
                    g_epoch == config.generator_pretrain_epoch - 1:
                _g_test_epoch(sess, g_epoch, 'valid')
                _g_test_epoch(sess, g_epoch, 'test')

        # Discriminator pre-training
        for d_epoch in range(config.discriminator_pretrain_epoch):
            _d_run_epoch(sess, d_epoch)

        # Adversarial training
        for update_epoch in range(config.adversial_epoch):
            cur_epoch = update_epoch + config.generator_pretrain_epoch
            _g_train_epoch(sess, cur_epoch, 'train')
            _d_run_epoch(sess, cur_epoch, mode_string='train')
            if update_epoch % 10 == 0 or \
                    update_epoch == config.adversial_epoch - 1:
                _g_test_epoch(sess, cur_epoch, 'valid')
                _g_test_epoch(sess, cur_epoch, 'test')

    log.close()
    bleu_log.close()

if __name__ == '__main__':
    tf.app.run(main=_main)

