"""
Trainer for classifier.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle as pkl
import numpy as np
import tensorflow as tf
import json
import os
import random


from texar.hyperparams import HParams
from texar.models.tsf import TSF
# from texar.models.tsf.classifier import Classifier
from texar.models.tsf import Classifier

from trainer_base import TrainerBase
from utils import *
from classifier_utils import *

flags = tf.flags
FLAGS = flags.FLAGS

class ClassifierTrainer(TrainerBase):
  """Classifier Trainer."""
  def __init__(self, hparams=None):
    TrainerBase.__init__(self, hparams)

  @staticmethod
  def default_hparams():
    return {
      "data_dir": "../../data/yelp",
      "expt_dir": "../../expt",
      "log_dir": "log",
      "name": "classifier",
      "batch_size": 128,
      "vocab_size": 10000,
      "gamma_init": 1,
      "gamma_decay": 0.5,
      "gamma_min": 0.1,
      "use_self_gate": False,
      "max_len": 20,
      "max_epoch": 20,
      "disp_interval": 100,
    }

  def prepare_data(self, train, val, test):
    train_x = train[0] + train[1]
    train_y = [0] * len(train[0]) + [1] * len(train[1])
    val_x = val[0] + val[1]
    val_y = [0] * len(val[0]) + [1] * len(val[1])
    test_x = test[0] + test[1]
    test_y = [0] * len(test[0]) + [1] * len(test[1])
    return (train_x, train_y), (val_x, val_y), (test_x, test_y)

  def eval_model(self, model, sess, vocab, x, y):
    batches = get_batches(x, y, vocab["word2id"], self._hparams.batch_size)
    probs = []
    losses = []
    for batch in batches:
      # loss, prob = model.eval_step(sess, batch,
      # gamma=self._hparams.gamma_min)
      loss, prob, alpha = model.visualize_step(
        sess, batch, self._hparams.gamma_min)
      losses += loss.tolist()[:batch["actual_size"]]
      probs += prob.tolist()[:batch["actual_size"]]
    y_hat = [ p > 0.5 for p in probs]
    same = [ p == q for p, q in zip(y, y_hat)]
    loss = sum(losses) / len(losses)
    accu = sum(same) / len(y)
    return loss, accu

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
    train, val, test = self.prepare_data(train, val, test)

    # set vocab size
    self._hparams.vocab_size = vocab["size"]

    with tf.Session() as sess:
      model = Classifier(self._hparams)
      log_print("finished building model")

      if "model" in self._hparams.keys():
        model.saver.restore(sess, self._hparams.model)
      else:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())


      gamma = self._hparams.gamma_init
      best_dev = float("-inf")
      loss = 0.
      accu = 0.
      step = 0
      batches = get_batches(train[0], train[1], vocab["word2id"],
                            self._hparams.batch_size, shuffle=True)
      
      log_dir = os.path.join(self._hparams.expt_dir, self._hparams.log_dir)
      train_writer = tf.summary.FileWriter(log_dir, sess.graph)

      for epoch in range(1, self._hparams.max_epoch + 1):
        # shuffle across batches
        log_print("------------------epoch %d --------------"%(epoch))
        log_print("gamma %.3f"%(gamma))
        random.shuffle(batches)

        for batch in batches:
          step_loss, step_accu = model.train_step(sess, batch, gamma)

          step += 1
          loss += step_loss / self._hparams.disp_interval
          accu += step_accu / self._hparams.disp_interval
          if step % self._hparams.disp_interval == 0:
            log_print("step %d: loss %.2f accu %.3f "%(step, loss, accu ))
            loss = 0.
            accu = 0.


        dev_loss, dev_accu = self.eval_model(model, sess, vocab, val[0],
                                             val[1])
        test_loss, test_accu = self.eval_model(model, sess, vocab, test[0],
                                               test[1])
        log_print("dev loss %.2f accu %.3f"%(dev_loss, dev_accu))
        log_print("test loss %.2f accu %.3f"%(test_loss, test_accu))
        if dev_accu > best_dev:
          best_dev = dev_accu
          file_name = (
            self._hparams["name"] + "_" + "%.3f" %(best_dev) + ".model")
          model.saver.save(
            sess, os.path.join(self._hparams['expt_dir'], file_name),
            latest_filename=self._hparams['name'] + '_checkpoint',
            global_step=step)
          log_print("saved model %s"%(file_name))

        gamma = max(self._hparams.gamma_min, gamma * self._hparams.gamma_decay)

    return best_dev

  def test(self):
    if "config" in self._hparams.keys():
      with open(self._hparams.config) as f:
        self._hparams = HParams(pkl.load(f), None)

    log_print(json.dumps(self._hparams.todict(), indent=2))

    vocab, train, val, test = self.load_data()
    train, val, test = self.prepare_data(train, val, test)

    # set vocab size
    self._hparams.vocab_size = vocab["size"]

    test_path = FLAGS.test
    if test_path.endswith(".pkl"):
      test = pkl.load(open(test_path))
      test_x = test[0] + test[1]
      test_y = [0] * len(test[0]) + [1] * len(test[1])
      test = (test_x, test_y)
    else:
      test_path_0 = test_path.replace("*", "0")
      test_path_1 = test_path.replace("*", "1")
      test_0 = load_data(test_path_0)
      test_0 = data_to_id(test_0, vocab["word2id"])
      test_1 = load_data(test_path_1)
      test_1 = data_to_id(test_1, vocab["word2id"])
      test_x = test_0 + test_1
      test_y = [0] * len(test_0) + [1] * len(test_1)
      test = (test_x, test_y)

    with tf.Session() as sess:
      model = Classifier(self._hparams)
      log_print("finished building model")

      model.saver.restore(sess, FLAGS.model)

      loss, accu = self.eval_model(model, sess, vocab, test[0],
                                   test[1])
      log_print("test loss %.2f accu %.3f"%(loss, accu))

    return accu

def main(unused_args):
  trainer = ClassifierTrainer()
  trainer.run()

if __name__ == "__main__":
  tf.app.run()










