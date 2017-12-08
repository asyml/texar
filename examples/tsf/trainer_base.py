"""
Base class for trainer.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb

import tensorflow as tf

from txtgen.hyperparams import HParams

flags = tf.flags
FLAGS = flags.FLAGS

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

    self._hparams = HParams(self._hparams, flags_hparams)

  @staticmethod
  def default_hparams():
    return {
    }

  def load_data(self):
    raise NotImplementedError

  def train():
    raise NotImplementedError

