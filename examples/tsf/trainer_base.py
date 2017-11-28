"""
Base class for trainer.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from txtgen.hyperparams import HParams

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("data_dir", "../../data/yelp", "data folder")
flags.DEFINE_string("expt_dir", "../../expt", "experiment folder")
flags.DEFINE_string("log_dir", "log", "experiment folder")
flags.DEFINE_integer("seed", 0, "random seed")
flags.DEFINE_string("config", "", "config to load.")
flags.DEFINE_string("model", "", "model to load.")

class TrainerBase(objec):
  """Base class for trainer."""
  def __init__(self, hparams=None):
    if FLAGS.data_dir:
      self._hparams['data_dir'] = FLAGS.data_dir
    if FLAGS.expt_dir:
      self._hparams['expt_dir'] = FLAGS.expt_dir
    if FLAGS.log_dir:
      self._hparams['log_dir'] = FLAGS.log_dir

    self._hparams = HParams(hparams, self.default_hparams())

  @staticmethod
  def default_hparams():
    return {
    }

  def load_data(self):
    raise NotImplementedError

  def train():
    raise NotImplementedError

