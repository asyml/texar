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
  def __init__(self, hparams=None):
    ModuleBase.__init__(self, hparams)

    if self._hparams.use_embedding:
      with tf.variable_scope(self.variable_scope):
        self._embedding = tf.get_variable("embedding",
        [self._hparams.vocab_size, self._hparams.embedding_size])
    if self._hparams.use_gate:
      self._gate_proj = tf.layers.Dense(self._hparams.attn_size)
      with tf.variable_scope(self.variable_scope):
        self._gate_u = tf.get_variable("u", [self._hparams.attn_size])
    self._conv_layers = []
    for k in self._hparams.kernel_sizes:
      conv_layer = tf.layers.Conv1D(self._hparams.num_filter, k,
                                    padding='same')
      self._conv_layers.append(conv_layer)

    self._proj_layer = tf.layers.Dense(1)

  @staticmethod
  def default_hparams():
    return {
      "name": "cnn",
      "use_embedding": False,
      "use_gate": False,
      "scale_attn": False,
      "attn_size": 100,
      "kernel_sizes": [3, 4, 5],
      "num_filter": 128,
      "output_keep_prob": 0.5,
      "input_keep_prob": 1.,
      "leaky_relu_alpha": 0.01,
      "vocab_size": 10000,
      "embedding_size": 100,
    }


  def _build(self, inputs, seq_len=None, gamma=None):
    if self._hparams.use_embedding:
      if inputs.get_shape().ndims == 2:
        inputs = tf.nn.embedding_lookup(self._embedding, inputs)
      elif inputs.get_shape().ndims == 3:
        inputs_shape = inputs.get_shape()
        inputs = tf.matmul(tf.reshape(inputs, [-1, inputs_shape[-1]]),
                           self._embedding)
        inputs = tf.reshape(inputs, [inputs_shape[0], -1, inputs.get_shape()[-1]])

    scores = tf.ones(tf.shape(inputs)[:2], tf.float32) \
             / tf.cast(tf.shape(inputs)[1], tf.float32)

    if seq_len is not None:
      mask = tf.sequence_mask(lengths=tf.to_int32(seq_len),
                              maxlen=tf.to_int32(tf.shape(inputs)[1]),
                              dtype=tf.float32)
    else:
      mask = tf.ones(tf.shape(inputs)[:2], tf.float32)


    if self._hparams.use_gate:
      proj = tf.tanh(self._gate_proj(inputs))
      if gamma is None:
        gamma = 1.
      scores = tf.reduce_sum(self._gate_u * proj, [2]) / gamma
      scores = scores * mask + ((1.0 - mask) * tf.float32.min)
      scores = tf.nn.softmax(scores)
      if self._hparams.scale_attn:
        scores = scores * tf.reduce_sum(mask, axis=1, keep_dims=True)
      inputs = tf.expand_dims(scores, 2) * inputs
    else:
      inputs = tf.expand_dims(mask, 2) * inputs


    # input keep prob??
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

    return logits, scores
