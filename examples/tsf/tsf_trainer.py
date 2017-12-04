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

from txtgen.hyperparams import HParams
from txtgen.models.tsf import TSF

from trainer_base import TrainerBase
from utils import *
from stats import Stats

class TSFTrainer(TrainerBase):
  """TSF trainer."""
  def __init__(self, hparams=None):
    self._hparams = HParams(hparams, self.default_hparams())
    TrainerBase.__init__(self, self._hparams)

  @staticmethod
  def default_hparams():
    return {
      "name": "tsf",
      "rho": 1.,
      "gamma_init": 1,
      "gamma_decay": 0.5,
      "gamma_min": 0.001,
      "disp_interval": 100,
      "batch_size": 128,
      "vocab_size": 10000,
      "max_len": 20,
      "max_epoch": 20,
      "sort_data": True,
      "shuffle_across_epoch": False
    }

  def load_data(self):
    hparams = self._hparams
    with open(os.path.join(hparams["data_dir"], "vocab.pkl")) as f:
      vocab = pkl.load(f)
    with open(os.path.join(hparams["data_dir"], "train.pkl")) as f:
      train = pkl.load(f)
    with open(os.path.join(hparams["data_dir"], "val.pkl")) as f:
      val = pkl.load(f)
    with open(os.path.join(hparams["data_dir"], "test.pkl")) as f:
      test = pkl.load(f)

    return vocab, train, val, test

  def eval_model(self, model, sess, vocab, data0, data1, output_path):
    batches, order0, order1 = get_batches(
      data0, data1, vocab["word2id"],
      self._hparams.batch_size, sort=self._hparams.sort_data)
    losses = Stats()

    data0_ori, data1_ori, data0_tsf, data1_tsf = [], [], [], []
    for batch in batches:
      logits_ori, logits_tsf = model.decode_step(sess, batch)

      loss, loss_g, ppl_g, loss_d, loss_d0, loss_d1 = model.eval_step(
        sess, batch, self._hparams.rho, self._hparams.gamma_min)
      batch_size = len(batch["enc_inputs"])
      word_size = np.sum(batch["weights"])
      losses.append(loss, loss_g, ppl_g, loss_d, loss_d0, loss_d1,
                    w_loss=batch_size, w_g=batch_size,
                    w_ppl=word_size, w_d=batch_size,
                    w_d0=batch_size, w_d1=batch_size)
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
    log_print(json.dumps(self._hparams.todict(), indent=2))
    if not "config" in self._hparams.keys():
      with open(os.path.join(self._hparams.expt_dir, self._hparams.name)
                + ".config", "w") as f:
        pkl.dump(self._hparams, f)

    vocab, train, val, test = self.load_data()

    # set vocab size
    self._hparams.vocab_size = vocab["size"]

    # set some hparams here

    with tf.Session() as sess:
      model = TSF(self._hparams)
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
      for epoch in range(self._hparams["max_epoch"]):
        # shuffle across batches
        if self._hparams.shuffle_across_epoch:
          batches = get_batches(train[0], train[1], vocab["word2id"],
                                model._hparams.batch_size,
                                sort=self._hparams.sort_data)
        random.shuffle(batches)
        for batch in batches:
          loss_d0 = model.train_d0_step(sess, batch, self._hparams.rho, gamma)
          loss_d1 = model.train_d1_step(sess, batch, self._hparams.rho, gamma)

          if loss_d0 < 1.2 and loss_d1 < 1.2:
            loss, loss_g, ppl_g, loss_d = model.train_g_step(
              sess, batch, self._hparams.rho, gamma)
          else:
            loss, loss_g, ppl_g, loss_d = model.train_ae_step(
              sess, batch, self._hparams.rho, gamma)

          losses.append(loss, loss_g, ppl_g, loss_d, loss_d0, loss_d1)

          step += 1
          if step % self._hparams.disp_interval == 0:
            log_print("step %d: "%(step) + str(losses))
            losses.reset()

        # eval on dev
        dev_loss = self.eval_model(
          model, sess, vocab, val[0], val[1],
          os.path.join(self._hparams.expt_dir,
                       "sentiment.dev.epoch%d"%(epoch)))
        log_print("dev " + str(dev_loss))
        if dev_loss.loss < best_dev:
          best_dev = dev_loss.loss
          file_name = (
            self._hparams['name'] + '_' + '%.2f' %(best_dev) + '.model')
          model.saver.save(
            sess, os.path.join(self._hparams['expt_dir'], file_name),
            latest_filename=self._hparams['name'] + '_checkpoint',
            global_step=step)
          log_print("saved model %s"%(file_name))

        gamma = max(self._hparams.gamma_min, gamma * self._hparams.gamma_decay)


def main(unused_args):
  trainer = TSFTrainer()
  trainer.train()

if __name__ == "__main__":
  tf.app.run()
