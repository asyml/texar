"""
Conv1D discriminator.
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


class CNN(ModuleBase):
  """CNN for text."""
  def __init__(self, hparams=None, use_embedding=False):
    ModuleBase.__init__(self, hparams)

    self._use_embedding = use_embedding
    if self._use_embedding:
      with self.variable_scope as scope:
        self._embedding = tf.get_variable("embedding",
        [self._hparams.vocab_size, self._hparams.embedding_size])
    self._conv_layers = []
    for k in self._hparams.kernel_sizes:
      conv_layer = tf.layers.Conv1D(self._hparams.num_filter, k)
      self._conv_layers.append(conv_layer)

    self._proj_layer = tf.layers.Dense(1)

  @staticmethod
  def default_hparams():
    return {
      "name": "cnn",
      "kernel_sizes": [3, 4, 5],
      "num_filter": 128,
      "output_keep_prob": 0.5,
      "input_keep_prob": 1,
      "leaky_relu_alpha": 0.01,
      "vocab_size": 10000,
      "embedding_size": 100,
    }


  def _build(self, inputs):
    if self._use_embedding:
      if inputs.get_shape().ndims == 2:
        inputs = tf.nn.embedding_lookup(self._embedding, inputs)
      elif inputs.get_shape().ndims == 3:
        inupts = tf.matmul(inputs, self._embedding)

    inputs = tf.nn.dropout(
      inputs, utils.switch_dropout(self._hparams.input_keep_prob))

    pooled_outputs = []
    for conv_layer in self._conv_layers:
      h = conv_layer(inputs)
      h = tf.nn.leaky_relu(h, alpha=self._hparams.leaky_relu_alpha)
      # pooling after conv
      h = tf.reduce_max(h, axis=1)
      pooled_outputs.append(h)

    outputs = tf.concat(pooled_outputs, 1)
    outputs = tf.nn.dropout(
      outputs, utils.switch_dropout(self._hparams.output_keep_prob))

    logits = self._proj_layer(outputs)

    self._add_internal_trainable_variables()
    self._built = True

    return logits
