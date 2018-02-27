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


def get_length(sample_id, end_token=2):
  if sample_id.get_shape().ndims == 3:
    sample_id = tf.argmax(sample_id, axis=-1, output_type=tf.int32)
  else:
    sample_id = tf.to_int32(sample_id)

  batch_size = sample_id.get_shape()[0]

  def condition(time, finished, sequence_lenghts):
    return tf.less(time, tf.shape(sample_id)[1])

  def loop(time, finished, sequence_lengths):
    next_finished = tf.equal(sample_id[:, time], end_token)
    next_finished = tf.logical_or(next_finished, finished)
    next_finished = tf.logical_or(next_finished, time + 1 >= tf.shape(sample_id)[1])
    next_sequence_lengths = tf.where(
      tf.logical_and(tf.logical_not(finished), next_finished),
      tf.fill(tf.shape(sequence_lengths), time+1),
      sequence_lengths)
    return (time + 1, next_finished, next_sequence_lengths)

  initial_time = tf.constant(0, dtype=tf.int32)
  initial_finished = tf.tile([False], [batch_size])
  initial_sequence_lengths = tf.zeros_like(initial_finished, dtype=tf.int32)
  res = tf.while_loop(
    condition,
    loop,
    loop_vars=[
      initial_time, initial_finished, initial_sequence_lengths,
    ]
  )
  return res[2]

def adv_loss(x_real, x_fake, discriminator, real_len=None, fake_len=None):
  real_logits, real_scores = discriminator(x_real, seq_len=real_len)
  real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    labels=tf.ones_like(real_logits), logits=real_logits))
  real_prob = tf.sigmoid(real_logits)
  real_pred = tf.cast(tf.greater(real_prob, 0.5), dtype=tf.int32)
  real_accu = tf.reduce_mean(tf.cast(tf.equal(real_pred, tf.ones_like(real_pred)),
                                     dtype=tf.float32))
  fake_logits, fake_scores = discriminator(x_fake, seq_len=fake_len)
  fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    labels=tf.zeros_like(fake_logits), logits=fake_logits))
  fake_prob = tf.sigmoid(fake_logits)
  fake_pred = tf.cast(tf.greater(fake_prob, 0.5), dtype=tf.int32)
  fake_accu = tf.reduce_mean(tf.cast(tf.equal(fake_pred, tf.zeros_like(fake_pred)),
                                     dtype=tf.float32))
  d_loss = real_loss + fake_loss
  accu = (real_accu + fake_accu) / 2.
  return d_loss, accu, real_scores, fake_scores

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



