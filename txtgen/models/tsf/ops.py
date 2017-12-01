"""
Some basic operations for text style transfer.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import pdb

import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import numpy as np

from txtgen.hyperparams import HParams
from txtgen.core import utils

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
      input_keep_prob=utils.switch_dropout(hparams["input_keep_prob"]),
      output_keep_prob=utils.switch_dropout(hparams["output_keep_prob"]),
      state_keep_prob=utils.switch_dropout(hparams["state_keep_prob"]))

    cells.append(cell)

  if hparams["num_layers"] > 1:
    cell = rnn.MultiRNNCell(cells)
  else:
    cell = cells[0]

  return cell

def gumbel_softmax(logits, gamma):
  eps = 1e-8
  u = tf.random_uniform(tf.shape(logits))
  g = -tf.log(-tf.log(u + eps) + eps)
  return tf.nn.softmax((logits + g) / gamma)

def feed_softmax(proj_layer, embedding, gamma):
  def loop_func(output):
    logits = proj_layer(output)
    prob = tf.nn.softmax(logits / gamma)
    inp = tf.matmul(prob, embedding)
    return inp, logits, None

  return loop_func

def sample_gumbel(proj_layer, embedding, gamma, straight_throught=False):
  def loop_func(output):
    logits = proj_layer(output)
    sample = gumbel_softmax(logits, gamma)
    if straight_throught:
      sample_hard = tf.cast(tf.equal(
        sample, tf.reduce_max(sample, 1, keep_dims=True)), tf.float32)
      sample = tf.stop_gradient(sample_hard - sample) + sample
    inp = tf.matmul(sample, embedding)
    return inp, logits, sample

  return loop_func

def greedy_softmax(proj_layer, embedding):
  def loop_func(output):
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
    sample_seq = tf.concat(sample, 1)
  else:
    sample_seq = None
  return output_seq, logits_seq, sample_seq

def adv_loss(x_real, x_fake, discriminator):
  real_logits = discriminator(x_real)
  real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    labels=tf.ones_like(real_logits), logits=real_logits))
  fake_logits = discriminator(x_fake)
  fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    labels=tf.zeros_like(fake_logits), logits=fake_logits))
  d_loss = real_loss + fake_loss
  return d_loss

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

