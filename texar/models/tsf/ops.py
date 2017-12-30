"""
Some basic operations for text style transfer.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import pdb
import collections

import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from tensorflow.contrib.distributions import RelaxedOneHotCategorical \
  as GumbelSoftmax


import numpy as np

from texar.hyperparams import HParams
from texar.core.utils import switch_dropout
from texar.models.tsf import utils

def get_rnn_cell(hparams=None):
  default_hparams = {
    "type": "BasicLSTMCell",
    "size": 256,
    "num_layers": 1,
    "input_keep_prob": 1.0,
    "output_keep_prob": 1.0,
    "state_keep_prob": 1.0,
  }

  hparams = HParams(hparams, default_hparams, allow_new_hparam=True)

  cells = []
  for i in range(hparams.num_layers):
    if hparams.type == "BasicLSTMCell":
      cell = rnn.BasicLSTMCell(hparams.size)
    else: # hparams.type == "GRU:
      cell = rnn.GRUCell(hparams.size)

    cell = rnn.DropoutWrapper(
      cell = cell,
      input_keep_prob=switch_dropout(hparams["input_keep_prob"]),
      output_keep_prob=switch_dropout(hparams["output_keep_prob"]),
      state_keep_prob=switch_dropout(hparams["state_keep_prob"]))

    cells.append(cell)

  if hparams["num_layers"] > 1:
    cell = rnn.MultiRNNCell(cells)
  else:
    cell = cells[0]

  return cell

def gumbel_softmax(gamma, logits=None, probs=None, straight_through=False):
  # if p is not None:
  #   logits = tf.log(p + 1e-8)
  # eps = 1e-8
  # u = tf.random_uniform(tf.shape(logits))
  # g = -tf.log(-tf.log(u + eps) + eps)
  # sample = tf.nn.softmax((logits + g) / gamma)
  sample = GumbelSoftmax(gamma, logits=logits, probs=probs).sample()
  if straight_through:
    sample_hard = tf.cast(tf.equal(
      sample, tf.reduce_max(sample, 1, keep_dims=True)), tf.float32)
    sample = tf.stop_gradient(sample_hard - sample) + sample
  return sample

def feed_softmax(proj_layer, embedding, gamma, output_keep_prob=0.5):
  def loop_func(output):
    output = tf.nn.dropout(output, switch_dropout(output_keep_prob))
    logits = proj_layer(output)
    prob = tf.nn.softmax(logits / gamma)
    inp = tf.matmul(prob, embedding)
    return inp, logits, prob

  return loop_func

def sample_gumbel(proj_layer, embedding, gamma, output_keep_prob=0.5,
                  straight_throught=False):
  def loop_func(output):
    output = tf.nn.dropout(output, switch_dropout(output_keep_prob))
    logits = proj_layer(output)
    sample = gumbel_softmax(gamma, logits=logits,
                            straight_through=straight_throught)
    inp = tf.matmul(sample, embedding)
    return inp, logits, sample

  return loop_func

def greedy_softmax(proj_layer, embedding, output_keep_prob=0.5):
  def loop_func(output):
    output = tf.nn.dropout(output, switch_dropout(output_keep_prob))
    logits = proj_layer(output)
    word = tf.argmax(logits, axis=1)
    inp = tf.nn.embedding_lookup(embedding, word)
    return inp, logits, word

  return loop_func

def rnn_decode(state, inp, length, cell, loop_func, scope):
  output_seq, logits_seq, sample_seq = [], [], []
  for t in range(length):
    # reuse cell params
    with tf.variable_scope(scope, reuse=True):
      output, state = cell(inp, state)
    inp, logits, sample = loop_func(output)
    output_seq.append(tf.expand_dims(output, 1))
    logits_seq.append(tf.expand_dims(logits, 1))
    if sample is not None:
      sample_seq.append(tf.expand_dims(sample, 1))
    else:
      sample_seq.append(sample)

  output_seq = tf.concat(output_seq, 1)
  logits_seq = tf.concat(logits_seq, 1)
  if sample[0] is not None:
    sample_seq = tf.concat(sample_seq, 1)
  else:
    sample_seq = None
  return output_seq, logits_seq, sample_seq

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


class PointerDecoder(object):
  def __init__(self, proj_layer, attn_layer, pointer_layer, enc_inputs,
               enc_outputs, embedding, output_keep_prob=0.5):
    self._proj_layer = proj_layer
    self._attn_layer = attn_layer
    self._pointer_layer = pointer_layer
    self._enc_inputs = enc_inputs
    self._enc_outputs = enc_outputs
    self._embedding = embedding
    self._output_keep_prob = output_keep_prob

  def compute_p(self, output):
    output = tf.nn.dropout(output, switch_dropout(self._output_keep_prob))
    attn_input = AttnInput(self._enc_outputs, output)
    attn_output = self._attn_layer(attn_input)
    p_attn = attn_output.p
    s_ave = attn_output.s_ave
    p_pointer = tf.sigmoid(self._pointer_layer(tf.concat([s_ave, output], 1)))
    # combine s_ave and output
    p_softmax = tf.nn.softmax(self._proj_layer(tf.conat([s_ave, output], 1)))
    vocab_size = p_softmax.get_shape()[-1]
    p_attn_dense = id_to_dense(p_attn, self._enc_inputs, vocab_size)
    p = p_pointer * p_attn_dense + (1 - p_pointer) * p_softmax
    debug_p = utils.register_collection("tsf/debug",
                                        [("p_attn", p_attn),
                                         ("p_attn_dense", p_attn_dense),
                                         ("p_pointer", p_pointer),
                                         ("p", p),
                                        ])
    # return p, debug_p
    return p_softmax, debug_p

  def next_inputs(self, t, p):
    raise NotImplementedError

class TrainPointerDecoder(PointerDecoder):
  def __init__(self, proj_layer, attn_layer, pointer_layer, enc_inputs,
               enc_outputs, embedding, dec_inputs, output_keep_prob=0.5):
    super(TrainPointerDecoder, self).__init__(
      proj_layer, attn_layer, pointer_layer, enc_inputs,
      enc_outputs, embedding)
    self._dec_inputs = dec_inputs

  def next_inputs(self, t, p):
    # next time step
    t = t + 1
    batch_size = self._dec_inputs.get_shape()[0]
    max_len = tf.shape(self._dec_inputs)[1]
    next_ids = tf.cond(t < max_len,
                       lambda: self._dec_inputs[:, t],
                       lambda: tf.zeros([batch_size], dtype=tf.int32))
    next_inps = tf.nn.embedding_lookup(self._embedding, next_ids)
    return next_inps, next_ids

class GumbelSoftmaxPointerDecoder(PointerDecoder):
  def __init__(self, proj_layer, attn_layer, pointer_layer, enc_inputs,
               enc_outputs, embedding, gamma, output_keep_prob=0.5,
               straight_through=False):
    super(GumbelSoftmaxPointerDecoder, self).__init__(
      proj_layer, attn_layer, pointer_layer, enc_inputs,
      enc_outputs, embedding)
    self._gamma = gamma
    self._straight_through = straight_through

  def next_inputs(self, t, p):
    sample = gumbel_softmax(self._gamma, probs=p,
                            straight_through=self._straight_through)
    next_inps = tf.matmul(sample, self._embedding)
    return next_inps, sample

class GreedyPointerDecoder(PointerDecoder):
  def __init__(self, proj_layer, attn_layer, pointer_layer, enc_inputs,
               enc_outputs, embedding, output_keep_prob=0.5):
    super(GreedyPointerDecoder, self).__init__(
      proj_layer, attn_layer, pointer_layer, enc_inputs,
      enc_outputs, embedding)
  def next_inputs(self, t, p):
    word = tf.argmax(p, axis=1)
    next_inps = tf.nn.embedding_lookup(self._embedding, word)
    return next_inps, word


class AttnInput(
    collections.namedtuple("AttnInput",
                           ("s", "q"))):
  pass

class AttnOutput(
    collections.namedtuple("AttnOutput",
                           ("p", "s_ave"))):
  pass

class AttnLayer(tf.layers.Layer):
  def __init__(self, size,
               activation=tf.tanh,
               trainable=True,
               name=None,
               **kwargs):
    super(AttnLayer, self).__init__(trainable=trainable, name=name, **kwargs)
    self.size = size
    self.activation = activation

  def build(self, input_shapes):
    input_shape_s = tf.TensorShape(input_shapes[0])
    input_shape_q = tf.TensorShape(input_shapes[1])
    if input_shape_s[-1].value is None or input_shape_q[-1].value is None:
      raise ValueError("The last dimension of inputs s and q should be defined.")
    self.w = self.add_variable(
      "w", shape=[input_shape_s[-1].value, self.size],
      dtype=self.dtype, trainable=True)
    self.u = self.add_variable(
      "u", shape=[input_shape_q[-1].value, self.size],
      dtype=self.dtype, trainable=True)
    self.v = self.add_variable(
      "v", shape=[self.size, 1], dtype=self.dtype, trainable=True)

    self.built = True

  def call(self, attn_input):
    s = attn_input.s
    q = attn_input.q
    s_shape = s.get_shape().as_list()
    q_shape = q.get_shape().as_list()
    sw = tf.tensordot(s, self.w, [[len(s_shape)-1], [0]])
    qu = tf.matmul(q, self.u)
    sq = self.activation(sw + tf.expand_dims(qu, 1))
    logits = tf.tensordot(sq, self.v, [[len(s_shape)-1], [0]])
    logits = tf.squeeze(logits, len(s_shape)-1)
    p = tf.nn.softmax(logits)
    s_ave = tf.reduce_sum(s * tf.expand_dims(p, 2), 1)
    attn_output = AttnOutput(p, s_ave)
    return attn_output

  def _compute_output_shape(self, input_shapes):
    s_shape = tf.Tensorshape(input_shapes[0])
    q_shape = tf.Tensorshape(input_shapes[0])
    return (s_shape[:2], s_shape[:1] + s_shape[-1:])

def rnn_pointer_decode(state, inp, length, cell, pointer_decoder,
                       scope="decoder", reuse=False):
  output_seq, p_seq, sample_seq = [], [], []
  p_attn_seq, p_attn_dense_seq, p_pointer_seq = [], [], []
  with tf.variable_scope(scope, reuse=reuse) as scope:
    for t in range(length):
      output, state = cell(inp, state)
      p, debug_p = pointer_decoder.compute_p(output)
      inp, sample= pointer_decoder.next_inputs(t, p)
      output_seq.append(tf.expand_dims(output, 1))
      p_seq.append(tf.expand_dims(p, 1))
      sample_seq.append(tf.expand_dims(sample, 1))
      p_attn_seq.append(tf.expand_dims(debug_p["p_attn"], 1))
      p_attn_dense_seq.append(tf.expand_dims(debug_p["p_attn_dense"], 1))
      p_pointer_seq.append(tf.expand_dims(debug_p["p_pointer"], 1))

  output_seq = tf.concat(output_seq, 1)
  p_seq = tf.concat(p_seq, 1)
  sample_seq = tf.concat(sample_seq, 1)
  p_attn_seq = tf.concat(p_attn_seq, 1)
  p_attn_dense_seq = tf.concat(p_attn_dense_seq, 1)
  p_pointer_seq = tf.concat(p_pointer_seq, 1)
  debug_p = utils.register_collection("tsf/debug",
                                      [("p_attn_seq", p_attn_seq),
                                       ("p_attn_dense_seq", p_attn_dense_seq),
                                       ("p_pointer_seq", p_pointer_seq),
                                       ("p", p),
                                      ])

  return output_seq, p_seq, sample_seq

def adv_loss(x_real, x_fake, discriminator):
  real_logits = discriminator(x_real)
  real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    labels=tf.ones_like(real_logits), logits=real_logits))
  real_prob = tf.sigmoid(real_logits)
  real_pred = tf.cast(tf.greater(real_prob, 0.5), dtype=tf.int32)
  real_accu = tf.reduce_mean(tf.cast(tf.equal(real_pred, tf.ones_like(real_pred)),
                                     dtype=tf.float32))
  fake_logits = discriminator(x_fake)
  fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    labels=tf.zeros_like(fake_logits), logits=fake_logits))
  fake_prob = tf.sigmoid(fake_logits)
  fake_pred = tf.cast(tf.greater(fake_prob, 0.5), dtype=tf.int32)
  fake_accu = tf.reduce_mean(tf.cast(tf.equal(fake_pred, tf.zeros_like(fake_pred)),
                                     dtype=tf.float32))
  d_loss = real_loss + fake_loss
  accu = (real_accu + fake_accu) / 2.
  return d_loss, accu

def retrieve_variables(scopes):
  var = []
  for scope in scopes:
    var += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
  return var

def feed_dict(model, batch, rho, gamma, dropout, learning_rate):
  return {
    dropout: context.is_train(),
    model.input_tensors["learning_rate"]: learning_rate,
    model.input_tensors["rho"]: rho,
    model.input_tensors["gamma"]: gamma,
    model.input_tensors["batch_len"]: batch["len"],
    model.input_tensors["enc_inputs"]: batch["enc_inputs"],
    model.input_tensors["dec_inputs"]: batch["dec_inputs"],
    model.input_tensors["targets"]: batch["targets"],
    model.input_tensors["weights"]: batch["weights"],
    model.input_tensprs["labels"]: batch["labels"],
  }

