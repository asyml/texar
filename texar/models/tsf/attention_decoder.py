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

def id_to_dense(p, word_id, vocab_size):
  batch_size = word_id.get_shape()[0]
  length = tf.shape(word_id)[1]
  idx = tf.expand_dims(tf.range(batch_size), 1)
  idx = tf.tile(idx, [1, length])
  idx = tf.stack([tf.reshape(idx, [-1]), tf.reshape(word_id, [-1])], axis=1)
  sparse_p = tf.SparseTensor(tf.cast(idx, tf.int64), tf.reshape(p, [-1]),
                             [batch_size, vocab_size])
  # word_id can have duplicate indices, ignored in the construction process
  batch_size = word_id.get_shape()[0]
  dense_p = tf.sparse_tensor_to_dense(sparse_p, validate_indices=False)
  return dense_p

class AttentionDecoderOutput(
    namedtuple("DecoderOutput", [
        "logits", "sample_id", "cell_output", "attention_scores",
        "attention_context"
    ])):
  """Augmented decoder output that also includes the attention scores.
  """
  pass

class PointerDecoderOutput(
    namedtuple("DecoderOutput", [
        "logits", "sample_id", "cell_output", "attention_scores",
        "attention_context", "pointer"
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
               output_layer=None,
               mask=None,
               hparams=None):
    super(AttentionDecoder, self).__init__(cell=cell,
                                           vocab_size=vocab_size,
                                           hparams=hparams)
    self._attention_keys = attention_keys
    self._attention_values = attention_values
    self._attention_values_length = attention_values_length
    self._attention_layer = attention_layer
    max_len = tf.shape(self._attention_values)[1]
    self._mask = mask
    if self._mask is not None:
      self._mask = self._mask[:, :max_len]
      _, self._mask = tf.nn.top_k(self._mask, self._hparams.mask_top_k)
      self._mask = tf.one_hot(self._mask, max_len, dtype=tf.float32)
      self._mask = 1. - tf.reduce_sum(self._mask, axis=1)
    with tf.variable_scope(self.variable_scope):
      # self._input_transform = tf.layers.Dense(, name="input_transform")
      self._softmax_input = tf.layers.Dense(self._cell.output_size,
                                            activation=tf.nn.tanh,
                                            name="softmax_input")
      if output_layer is None:
        self._output_layer = tf.layers.Dense(vocab_size, name="output_layer")
      else:
        self._output_layer = output_layer

  def set_attention_inputs(self, attention_keys, attention_values,
                           attention_values_length):
    self._attention_keys = attention_keys
    self._attention_values = attention_values
    self._attention_values_length = attention_values_length

  @staticmethod
  def default_hparams():
    hparams = RNNDecoderBase.default_hparams()
    hparams["name"] = "attention_decoder"
    hparams["use_embedding"] = False
    hparams["mask_top_k"] = 3
    return hparams

  @property
  def output_size(self):
    return AttentionDecoderOutput(
      logits=self._vocab_size,
      sample_id=self._helper.sample_ids_shape,
      cell_output=self._cell.output_size,
      attention_scores=tf.shape(self._attention_values)[1:-1],
      attention_context=self._attention_values.get_shape()[-1])

  @property
  def output_dtype(self):
    return AttentionDecoderOutput(
      logits=tf.float32,
      sample_id=self._helper.sample_ids_dtype,
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
      values_length=self._attention_values_length,
      mask=self._mask)

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
      sample_id=sample_ids,
      cell_output=cell_output_new,
      attention_scores=att_scores,
      attention_context=att_context)

    finished, next_inputs, next_state = self._helper.next_inputs(
      time=time_, outputs=outputs, state=cell_state, sample_ids=sample_ids)

    # next_inputs = tf.cocnat([next_inputs, att_context], 1)

    return (outputs, next_state, next_inputs, finished)

  def finalize(self, outputs, final_state, sequence_lengths):
    return outputs, final_state


class PointerDecoder(AttentionDecoder):
  def __init__(self, vocab_size,
               attention_keys,
               attention_values,
               attention_values_length,
               attention_layer,
               cell,
               input_ids,
               hparams=None):
    super(PointerDecoder, self).__init__(
      vocab_size,
      attention_keys,
      attention_values,
      attention_values_length,
      attention_layer,
      cell,
      hparams)
    self._input_ids = input_ids
    with tf.variable_scope(self.variable_scope):
      self._pointer_layer = tf.layers.Dense(1, name="pointer_layer")

  @staticmethod
  def default_hparams():
    hparams = AttentionDecoder.default_hparams()
    hparams["name"] = "pointer_decoder"
    hparams["use_embedding"] = False

    return hparams

  @property
  def output_size(self):
    return PointerDecoderOutput(
      logits=self._vocab_size,
      sample_id=self._helper.sample_ids_shape,
      cell_output=self._cell.output_size,
      attention_scores=tf.shape(self._attention_values)[1:-1],
      attention_context=self._attention_values.get_shape()[-1],
      pointer=1)

  @property
  def output_dtype(self):
    return PointerDecoderOutput(
      logits=tf.float32,
      sample_id=self._helper.sample_ids_dtype,
      cell_output=tf.float32,
      attention_scores=tf.float32,
      attention_context=tf.float32,
      pointer=tf.float32)

  def _compute_output(self, cell_output):
    att_scores, att_context = self._attention_layer(
      query=cell_output,
      keys=self._attention_keys,
      values=self._attention_values,
      values_length=self._attention_values_length)

    softmax_input = self._softmax_input(
      tf.concat([cell_output, att_context], 1))
    pointer = tf.sigmoid(self._pointer_layer(softmax_input))
    logits = self._output_layer(softmax_input)
    prob = tf.nn.softmax(logits)
    p_attn_dense = id_to_dense(att_scores, self._input_ids, self._vocab_size)
    p_sum = pointer * p_attn_dense + (1. - pointer) * prob
    logits = tf.log(p_sum + 1e-8)
    return softmax_input, logits, att_scores, att_context, pointer

  def step(self, time_, inputs, state, name=None):
    cell_output, cell_state = self._cell(inputs, state)
    cell_output_new, logits, att_scores, att_context, pointer,\
      = self._compute_output(cell_output)

    sample_ids = self._helper.sample(
      time=time_, outputs=logits, state=cell_state)

    outputs = PointerDecoderOutput(
      logits=logits,
      sample_id=sample_ids,
      cell_output=cell_output_new,
      attention_scores=att_scores,
      attention_context=att_context,
      pointer=pointer)

    finished, next_inputs, next_state = self._helper.next_inputs(
      time=time_, outputs=outputs, state=cell_state, sample_ids=sample_ids)

    # next_inputs = tf.cocnat([next_inputs, att_context], 1)

    return (outputs, next_state, next_inputs, finished)
