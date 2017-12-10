"""
Self Gate
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb

import tensorflow as tf

from texar.hyperparams import HParams
from texar.modules.module_base import ModuleBase
from texar.core.layers import get_layer
from texar.core import utils


class SelfGate(ModuleBase):
  """Self Gate."""
  def __init__(self, hparams=None):
    ModuleBase.__init__(self, hparams)

    if self._hparams.type == "Attn":
      self._proj_layer = tf.layers.Dense(self._hparams.size)
      self._u = tf.get_variable("u", [self._hparams.size, 1])
    elif self._hparams.type == "Gate":
      self._hidden_layer = tf.layers.Dense(self._hparams.size)
      self._proj_layer = tf.layers.Dense(1)

  @staticmethod
  def default_hparams():
    return {
      "name": "self_gate",
      # Gate and Attn
      "type": "Attn",
      "size": 100,
      "leaky_relu_alpha": 0.2
    }


  def _build(self, inputs, gamma):
    batch_size = inputs.get_shape().as_list()[0]
    input_dim = inputs.get_shape().as_list()[-1]
    inputs_flat = tf.reshape(inputs, [-1, input_dim])

    if self._hparams.type == "Attn":
      proj = self._proj_layer(inputs_flat)
      proj = tf.nn.leaky_relu(proj, alpha=self._hparams.leaky_relu_alpha)
      prod = tf.matmul(proj, self._u)
      prod = tf.reshape(prod, [batch_size, -1])
      alpha = tf.nn.softmax(prod / gamma)
      output = tf.expand_dims(alpha, 2) * inputs
    elif self._hparams.type == "Gate":
      h = self._hidden_layer(inputs_flat)
      # h = tf.nn.leaky_relu(h, alpha=self._hparams.leaky_relu_alpha)
      t = tf.tanh(h)
      proj = self._proj_layer(h)
      proj = tf.reshape(proj, [batch_size, -1])
      alpha = tf.sigmoid(proj)
      output = tf.expand_dims(alpha, 2) * inputs

    self._add_internal_trainable_variables()
    self._built = True

    return output, alpha
