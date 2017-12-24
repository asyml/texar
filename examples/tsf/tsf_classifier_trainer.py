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

from texar.hyperparams import HParams
from texar.models.tsf import TSFClassifier

from trainer_base import TrainerBase
from utils import *
from tsf_utils import *
from stats import TSFClassifierStats as Stats

class TSFClassifierTrainer(TrainerBase):
  """TSFClassifier trainer."""
  def __init__(self, hparams=None):
    TrainerBase.__init__(self, hparams)

  @staticmethod
  def default_hparams():
    return {
      "data_dir": "../../data/yelp",
      "expt_dir": "../../expt",
      "log_dir": "log",
      "name": "tsf",
      "rho_f": 1.,
      "rho_r": 0.,
      "gamma_init": 1,
      "gamma_decay": 0.5,
      "gamma_min": 0.001,
      "disp_interval": 100,
      "batch_size": 128,
      "vocab_size": 10000,
      "cnn_vocab_size": 10000,
      "max_len": 20,
      "max_epoch": 20,
      "sort_data": False,
      "shuffle_across_epoch": True,
      "d_update_freq": 1,
    }


  def eval_model(self, model, sess, vocab, data0, data1, output_path):
    batches, order0, order1 = get_batches(
      data0, data1, vocab["word2id"],
      self._hparams.batch_size, sort=self._hparams.sort_data)
    losses = Stats()

    data0_ori, data1_ori, data0_tsf, data1_tsf = [], [], [], []
    for batch in batches:
      logits_ori, logits_tsf = model.decode_step(sess, batch)

      loss, loss_g, ppl_g, loss_df, loss_dr, loss_ds, \
        accu_f, accu_r, accu_s = model.eval_step(
        sess, batch, self._hparams.rho_f, self._hparams.rho_r,
        self._hparams.gamma_min)
      batch_size = len(batch["enc_inputs"])
      word_size = np.sum(batch["weights"])
      losses.append(loss, loss_g, ppl_g, loss_df, loss_dr, loss_ds,
                    accu_f, accu_r, accu_s, 
                    w_loss=batch_size, w_g=batch_size,
                    w_ppl=word_size, w_df=batch_size,
                    w_dr=batch_size, w_ds=batch_size,
                    w_af=batch_size, w_ar=batch_size,
                    w_as=batch_size)
      ori = logits2word(logits_ori, vocab["id2word"])
      tsf = logits2word(logits_tsf, vocab["id2word"])
      half = self._hparams.batch_size // 2
      data0_ori += ori[:half]
      data1_ori += ori[half:]
      data0_tsf += tsf[:half]
      data1_tsf += tsf[half:]

    n0 = len(data0)
    n1 = len(data1)
    data0_ori = reorder(order0, data0_ori)[:n0]
    data1_ori = reorder(order1, data1_ori)[:n1]
    data0_tsf = reorder(order0, data0_tsf)[:n0]
    data1_tsf = reorder(order1, data1_tsf)[:n1]

    write_sent(data0_ori, output_path + ".0.ori")
    write_sent(data1_ori, output_path + ".1.ori")
    write_sent(data0_tsf, output_path + ".0.tsf")
    write_sent(data1_tsf, output_path + ".1.tsf")
    return losses

  def train(self):
    if "config" in self._hparams.keys():
      with open(self._hparams.config) as f:
        self._hparams = HParams(pkl.load(f))

    log_print("Start training with hparams:")
    log_print(str(self._hparams))
    if not "config" in self._hparams.keys():
      with open(os.path.join(self._hparams.expt_dir, self._hparams.name)
                + ".config", "w") as f:
        pkl.dump(self._hparams, f)

    vocab, train, val, test = self.load_data()

    # set vocab size
    self._hparams.vocab_size = vocab["size"]
    self._hparams.cnn_vocab_size = vocab["size"]

    # set some hparams here

    with tf.Session() as sess:
      model = TSFClassifier(self._hparams)
      log_print("finished building model")

      if "model" in self._hparams.keys():
        model.saver.restore(sess, self._hparams.model)
      else:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

      losses = Stats()
      gamma = self._hparams.gamma_init
      step = 0
      best_dev = float("inf")
      batches, _, _ = get_batches(train[0], train[1], vocab["word2id"],
                                  model._hparams.batch_size,
                                  sort=self._hparams.sort_data)

      log_dir = os.path.join(self._hparams.expt_dir, self._hparams.log_dir)
      train_writer = tf.summary.FileWriter(log_dir, sess.graph)

      for epoch in range(1, self._hparams["max_epoch"] + 1):
        # shuffle across batches
        log_print("------------------epoch %d --------------"%(epoch))
        log_print("gamma %.3f"%(gamma))
        if self._hparams.shuffle_across_epoch:
          batches, _, _ = get_batches(train[0], train[1], vocab["word2id"],
                                      model._hparams.batch_size,
                                      sort=self._hparams.sort_data)
        random.shuffle(batches)
        for batch in batches:
          loss_ds = 0.
          for _ in range(self._hparams.d_update_freq):
            loss_ds, accu_s = model.train_d_step(sess, batch)

          if loss_ds < 1.2:
            (loss, loss_g, ppl_g, loss_df, loss_dr,
             accu_f, accu_r) = model.train_g_step(
               sess, batch, self._hparams.rho_f, self._hparams.rho_r, gamma)
          else:
            (loss, loss_g, ppl_g, loss_df, loss_dr,
             accu_f, accu_r)= model.train_ae_step(
               sess, batch, self._hparams.rho_f, self._hparams.rho_r, gamma)

          losses.append(loss, loss_g, ppl_g, loss_df, loss_dr, loss_ds,
                        accu_f, accu_r, accu_s)

          step += 1
          if step % self._hparams.disp_interval == 0:
            log_print("step %d: "%(step) + str(losses))
            losses.reset()

        # eval on dev
        dev_loss = self.eval_model(model, sess, vocab, val[0], val[1],
          os.path.join(log_dir, "sentiment.dev.epoch%d"%(epoch)))
        log_print("dev " + str(dev_loss))
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
  trainer = TSFClassifierTrainer()
  trainer.train()

if __name__ == "__main__":
  tf.app.run()
