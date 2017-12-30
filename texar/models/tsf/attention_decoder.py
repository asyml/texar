""" Implementations of attention layers.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import pdb

from collections import namedtuple

import tensorflow as tf

from texar.modules.decoders.rnn_decoder_base import RNNDecoderBase
from texar.models.tsf import AttentionLayerBahdanau

class AttentionDecoderOutput(
    namedtuple("DecoderOutput", [
        "logits", "predicted_ids", "cell_output", "attention_scores",
        "attention_context"
    ])):
  """Augmented decoder output that also includes the attention scores.
  """
  pass

class AttentionDecoder(RNNDecoderBase):
  def __init__(self, vocab_size,
               attention_keys,
               attention_values,
               attention_values_length,
               attention_layer,
               cell,
               hparams=None):
    super(AttentionDecoder, self).__init__(cell=cell,
                                           vocab_size=vocab_size,
                                           hparams=hparams)
    self._attention_keys = attention_keys
    self._attention_values = attention_values
    self._attention_values_length = attention_values_length
    self._attention_layer = attention_layer
    with tf.variable_scope(self.variable_scope):
      # self._input_transform = tf.layers.Dense(, name="input_transform")
      self._softmax_input = tf.layers.Dense(self._cell.output_size,
                                            activation=tf.nn.tanh,
                                            name="softmax_input")
      self._output_layer = tf.layers.Dense(vocab_size, name="output_layer")

  @staticmethod
  def default_hparams():
    hparams = RNNDecoderBase.default_hparams()
    hparams["name"] = "attention_decoder"
    hparams["use_embedding"] = False

    return hparams

  @property
  def output_size(self):
    return AttentionDecoderOutput(
      logits=self._vocab_size,
      predicted_ids=self._helper.sample_ids_shape,
      cell_output=self._cell.output_size,
      attention_scores=tf.shape(self._attention_values)[1:-1],
      attention_context=self._attention_values.get_shape()[-1])

  @property
  def output_dtype(self):
    return AttentionDecoderOutput(
      logits=tf.float32,
      predicted_ids=self._helper.sample_ids_dtype,
      cell_output=tf.float32,
      attention_scores=tf.float32,
      attention_context=tf.float32)

  def initialize(self, name=None):
    finished, first_inputs = self._helper.initialize()
    attention_context = tf.zeros([
      tf.shape(first_inputs)[0],
      self._attention_values.get_shape().as_list()[-1]
    ])
    # first_inputs = tf.concat([first_inputs, attention_context], 1)

    return finished, first_inputs, self._initial_state

  def _compute_output(self, cell_output):
    att_scores, att_context = self._attention_layer(
      query=cell_output,
      keys=self._attention_keys,
      values=self._attention_values,
      values_length=self._attention_values_length)

    softmax_input = self._softmax_input(
      tf.concat([cell_output, att_context], 1))
    logits = self._output_layer(softmax_input)
    return softmax_input, logits, att_scores, att_context

  def step(self, time_, inputs, state, name=None):
    cell_output, cell_state = self._cell(inputs, state)
    cell_output_new, logits, att_scores, att_context \
      = self._compute_output(cell_output)

    sample_ids = self._helper.sample(
      time=time_, outputs=logits, state=cell_state)

    outputs = AttentionDecoderOutput(
      logits=logits,
      predicted_ids=sample_ids,
      cell_output=cell_output_new,
      attention_scores=att_scores,
      attention_context=att_context)

    finished, next_inputs, next_state = self._helper.next_inputs(
      time=time_, outputs=outputs, state=cell_state, sample_ids=sample_ids)

    # next_inputs = tf.cocnat([next_inputs, att_context], 1)

    return (outputs, next_state, next_inputs, finished)

  def finalize(self, outputs, final_state, sequence_lengths):
    return outputs, final_state

