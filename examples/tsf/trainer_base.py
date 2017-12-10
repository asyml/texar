"""
Base class for trainer.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb

import cPickle as pkl
import os
import tensorflow as tf

from texar.hyperparams import HParams

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("test", "", "test path")
flags.DEFINE_string("data_dir", "", "data folder")
flags.DEFINE_string("expt_dir", "", "experiment folder")
flags.DEFINE_string("log_dir", "", "experiment folder")
flags.DEFINE_integer("seed", 0, "random seed")
flags.DEFINE_string("config", "", "config to load.")
flags.DEFINE_string("model", "", "model to load.")

class TrainerBase(object):
  """Base class for trainer."""
  def __init__(self, hparams=None):
    flags_hparams = self.default_hparams()
    if FLAGS.data_dir:
      flags_hparams["data_dir"] = FLAGS.data_dir
    if FLAGS.expt_dir:
      flags_hparams["expt_dir"] = FLAGS.expt_dir
    if FLAGS.log_dir:
      flags_hparams["log_dir"] = FLAGS.log_dir
    if FLAGS.config:
      flags_hparams["config"] = FLAGS.config
    if FLAGS.model:
      flags_hparams["model"] = FLAGS.model

    self._hparams = HParams(hparams, flags_hparams)

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

  @staticmethod
  def default_hparams(self):
    raise NotImplementedError

  def train(self):
    raise NotImplementedError

  def test(self):
    raise NotImplementedError

  def run(self):
    if FLAGS.test:
      self.test()
    else:
      self.train()
