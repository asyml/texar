"""
Trainer for tsf.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb

import cPickle as pkl
import numpy as np
import tensorflow as tf
import json
import os

from utils import *
import texar as tx
from texar.data import SpecialTokens
from texar.hyperparams import HParams
from texar.models.tsf import TSF

flags = tf.flags
FLAGS = flags.FLAGS



flags.DEFINE_string("expt_dir", "", "experiment dir")
flags.DEFINE_string("log_dir", "", "log dir")
flags.DEFINE_string("config", "", "config")
flags.DEFINE_string("model", "", "model")

class TSFTrainer:
    """TSF trainer."""
    def __init__(self, hparams=None):
        flags_hparams = self.default_hparams()
        if FLAGS.expt_dir:
            flags_hparams["expt_dir"] = FLAGS.expt_dir
        if FLAGS.log_dir:
            flags_hparams["log_dir"] = FLAGS.log_dir
        if FLAGS.config:
            flags_hparams["config"] = FLAGS.config
        if FLAGS.model:
            flags_hparams["model"] = FLAGS.model

        self._hparams = HParams(hparams, flags_hparams, allow_new_hparam=True)

    @staticmethod
    def default_hparams():
        return {
            "train_data_hparams": {
                "batch_size": 64,
                "num_epochs": 1,
                "allow_smaller_final_batch": False,
                "bucket_boundaries": np.arange(2, 20, 2).tolist(),
                "bucket_length_fn": "len_pair",
                "source_dataset": {
                    "files": "../../data/yelp/sentiment.train.sort.0",
                    "vocab_file": "../../data/yelp/vocab",
                    "bos_token": SpecialTokens.BOS,
                    "eos_token": SpecialTokens.EOS,
                },
                "target_dataset": {
                    "files": "../../data/yelp/sentiment.train.sort.1",
                    "vocab_share": True,
                },
            },
            "val_data_hparams": {
                "batch_size": 64,
                "num_epochs": 1,
                "shuffle": False,
                "allow_smaller_final_batch": False,
                "source_dataset": {
                    "files": "../../data/yelp/sentiment.dev.sort.0",
                    "vocab_file": "../../data/yelp/vocab",
                    "bos_token": SpecialTokens.BOS,
                    "eos_token": SpecialTokens.EOS,
                },
                "target_dataset": {
                    "files": "../../data/yelp/sentiment.dev.sort.1",
                    "vocab_share": True,
                },
            },
            "test_data_hparams": {
                "batch_size": 64,
                "num_epochs": 1,
                "shuffle": False,
                "allow_smaller_final_batch": False,
                "source_dataset": {
                    "files": "../../data/yelp/sentiment.test.sort.0",
                    "vocab_file": "../../data/yelp/vocab",
                    "bos_token": SpecialTokens.BOS,
                    "eos_token": SpecialTokens.EOS,
                },
                "target_dataset": {
                    "files": "../../data/yelp/sentiment.test.sort.1",
                    "vocab_share": True,
                },
            },
            "vocab_size": 10000,
            "batch_size": 128,
            "expt_dir": "../../expt",
            "log_dir": "log",
            "name": "tsf",
            "rho_adv": 0.,
            "rho_f": 0.5,
            "gamma_init": 1,
            "gamma_decay": 0.5,
            "gamma_min": 0.001,
            "disp_interval": 100,
            "max_epoch": 20,
        }


    def eval_model(self, model, sess, dataset, iterator, input_tensors,
                   output_path):
        losses = Stats()
        id2word = dataset.vocab[0].id_to_token_map_py

        data0_ori, data1_ori, data0_tsf, data1_tsf = [], [], [], []
        while True:
            try:
                batch = sess.run(input_tensors,
                                 {tx.global_mode(): tf.estimator.ModeKeys.EVAL})
                logits_ori, logits_tsf = model.decode_step(sess, batch)
                loss, loss_g, ppl_g, loss_d, loss_d0, loss_d1,\
                    loss_ds, loss_df = model.eval_step(
                        sess, batch, self._hparams.gamma_min)

                batch_size = batch["enc_inputs"].shape[0]
                word_size = np.sum(batch["seq_len"])
                losses.append(loss, loss_g, ppl_g, loss_d, loss_d0, loss_d1,
                              loss_ds, loss_df, 
                              w_loss=batch_size, w_g=batch_size,
                              w_ppl=word_size, w_d=batch_size,
                              w_d0=batch_size, w_d1=batch_size,
                              w_ds=batch_size, w_df=batch_size)
                ori = logits2word(logits_ori, id2word)
                tsf = logits2word(logits_tsf, id2word)
                half = self._hparams.batch_size // 2
                data0_ori += ori[:half]
                data1_ori += ori[half:]
                data0_tsf += tsf[:half]
                data1_tsf += tsf[half:]
            except tf.errors.OutOfRangeError:
                break

        n = dataset._dataset_size
        data0_ori = data0_ori[:n]
        data1_ori = data1_ori[:n]
        data0_tsf = data0_tsf[:n]
        data1_tsf = data1_tsf[:n]

        write_sent(data0_ori, output_path + ".0.ori")
        write_sent(data1_ori, output_path + ".1.ori")
        write_sent(data0_tsf, output_path + ".0.tsf")
        write_sent(data1_tsf, output_path + ".1.tsf")
        return losses

    def preprocess_input(self, data_batch):
        src = data_batch["source_text_ids"]
        src_len = data_batch["source_length"]
        tgt = data_batch["target_text_ids"]
        tgt_len = data_batch["target_length"]
        l = tf.maximum(tf.shape(src)[1], tf.shape(tgt)[1])
        batch_size = tf.shape(src)[0]
        # padding
        src = tf.pad(src, [[0, 0], [0, l - tf.shape(src)[1]]])
        tgt= tf.pad(tgt, [[0, 0], [0, l - tf.shape(tgt)[1]]])
        # concatenate
        inputs = tf.concat([src, tgt], axis=0)
        inputs_len = tf.concat([src_len, tgt_len], axis=0)
        enc_inputs = tf.reverse_sequence(inputs, inputs_len, seq_dim=1)
        # remove EOS
        enc_inputs = enc_inputs[:, 1:]
        enc_inputs = tf.reverse_sequence(enc_inputs, inputs_len - 1, seq_dim=1)
        dec_inputs = enc_inputs
        enc_inputs = dec_inputs[:, 1:]
        enc_inputs = tf.reverse_sequence(enc_inputs, inputs_len - 2, seq_dim=1)
        targets = inputs[:, 1:]
        labels = tf.concat([tf.zeros([batch_size]),
                            tf.ones([batch_size])],
                           axis=0)
        return {
            "enc_inputs": enc_inputs,
            "dec_inputs": dec_inputs,
            "targets": targets,
            "seq_len": inputs_len - 1,
            "labels": labels,
        }

    def train(self):
        if "config" in self._hparams.keys():
            with open(self._hparams.config) as f:
                self._hparams = HParams(pkl.load(f), None)

        log_print("Start training with hparams:")
        log_print(json.dumps(self._hparams.todict(), indent=2))
        if not "config" in self._hparams.keys():
            with open(os.path.join(self._hparams.expt_dir, self._hparams.name)
                      + ".config", "w") as f:
                pkl.dump(self._hparams, f)

        train_data = tx.data.PairedTextData(self._hparams.train_data_hparams)
        val_data = tx.data.PairedTextData(self._hparams.val_data_hparams)
        test_data = tx.data.PairedTextData(self._hparams.test_data_hparams)
        iterator = tx.data.TrainTestDataIterator(train=train_data,
                                                 val=val_data, test=test_data)
        data_batch = iterator.get_next()
        input_tensors = self.preprocess_input(data_batch)

        self._hparams.vocab_size = train_data._src_vocab.size

        with tf.Session() as sess:
            losses = Stats()
            model = TSF(self._hparams)
            if FLAGS.model:
                model.saver.restore(sess, FLAGS.model)
            else:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
            sess.run(tf.tables_initializer())

            log_print("finished building model")

            step = 0
            best_dev = float("inf")
            gamma = self._hparams.gamma_init
            log_dir = os.path.join(self._hparams.expt_dir, self._hparams.log_dir)
            train_writer = tf.summary.FileWriter(log_dir, sess.graph)

            for epoch in range(1, self._hparams["max_epoch"] + 1):
                log_print("------------------epoch %d --------------"%(epoch))
                log_print("gamma %.3f"%(gamma))

                # one epoch
                iterator.switch_to_train_data(sess)
                while True:
                    try:
                        batch = sess.run(
                            input_tensors,
                            {tx.global_mode(): tf.estimator.ModeKeys.EVAL})
                        loss_ds = model.train_ds_step(sess, batch, gamma)
                        loss_d0 = model.train_d0_step(sess, batch, gamma)
                        loss_d1 = model.train_d1_step(sess, batch, gamma)

                        if loss_ds < 1.2 or (loss_d0 < 1.2 and loss_d1 < 1.2):
                            try:
                                loss, loss_g, ppl_g, loss_d, loss_df \
                                    = model.train_g_step(sess, batch, gamma)
                            except:
                                print(batch)
                                print(batch["seq_len"])
                                print(batch["dec_inputs"].shape)
                                print(batch["targets"].shape)
                        else:
                            loss, loss_g, ppl_g, loss_d, loss_df \
                                = model.train_ae_step(sess, batch, gamma)

                        losses.append(loss, loss_g, ppl_g, loss_d, loss_d0, loss_d1,
                                      loss_ds, loss_df)

                        step += 1
                        if step % self._hparams.disp_interval == 0:
                            log_print("step %d: "%(step) + str(losses))
                            losses.reset()
                    except tf.errors.OutOfRangeError:
                        break

                # eval on dev
                iterator.switch_to_val_data(sess)
                dev_loss = self.eval_model(
                    model, sess, val_data, iterator, input_tensors,
                    os.path.join(log_dir, "sentiment.dev.epoch%d"%(epoch)))
                log_print("dev " + str(dev_loss))

                iterator.switch_to_test_data(sess)
                test_loss = self.eval_model(
                    model, sess, test_data, iterator, input_tensors,
                    os.path.join(log_dir, "sentiment.test.epoch%d"%(epoch)))
                log_print("test " + str(test_loss))

                if dev_loss.loss < best_dev:
                    best_dev = dev_loss.loss
                    file_name = (
                        self._hparams["name"] + "_" + "%.2f" %(best_dev) + ".model")
                    model.saver.save(
                        sess, os.path.join(self._hparams["expt_dir"], file_name),
                        latest_filename=self._hparams["name"] + "_checkpoint",
                        global_step=step)
                    log_print("saved model %s"%(file_name))

                gamma = max(self._hparams.gamma_min, gamma * self._hparams.gamma_decay)
                
            return best_dev

def main(unused_args):
    trainer = TSFTrainer()
    trainer.train()

if __name__ == "__main__":
    tf.app.run()
