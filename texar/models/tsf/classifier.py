"""
Text Classifier.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb

import numpy as np
import tensorflow as tf

from texar import context
from texar.hyperparams import HParams
from texar.core.utils import switch_dropout
from texar.modules.encoders.conv1d_discriminator import CNN
from texar.modules.encoders.self_gate import SelfGate
from texar.models.tsf import ops
from texar.models.tsf import utils


class Classifier:
  def __init__(self, hparams=None):
    self._hparams = HParams(hparams, self.default_hparams(),
                            allow_new_hparam=True)
    self.tensors = self._build_model()
    self.saver = tf.train.Saver()

  @staticmethod
  def default_hparams():
    return {
      "name": "classifier",
      "collections": "classifier",
      "batch_size": 128,
      "embedding_size": 100,
      # cnn
      "cnn_name": "cnn",
      "cnn_use_embedding": True,
      "cnn_use_gate": False,
      "cnn_kernel_sizes": [3, 4, 5],
      "cnn_num_filter": 128,
      "cnn_input_keep_prob": 1.,
      "cnn_output_keep_prob": 0.5,
      # adam
      "adam_learning_rate": 1e-4,
      "adam_beta1": 0.9,
      "adam_beta2": 0.999
    }

  def _build_model(self):
    hparams = self._hparams
    batch_size = hparams.batch_size
    cnn_hparams = utils.filter_hparams(hparams, "cnn")
    adam_hparams = utils.filter_hparams(hparams, "adam")

    x = tf.placeholder(tf.int32, [batch_size, None], name="x")
    y = tf.placeholder(tf.int32, [batch_size], name="y")
    gamma = tf.placeholder(tf.float32, [1], name="gamma")

    cnn = CNN(cnn_hparams)
    logits, scores = cnn(x)
    logits = tf.reshape(logits, [-1])
    prob = tf.sigmoid(logits)
    pred = tf.cast(tf.greater(prob, 0.5), dtype=tf.int32)
    accu = tf.reduce_mean(tf.cast(tf.equal(pred, y), dtype=tf.float32))

    losses = tf.nn.sigmoid_cross_entropy_with_logits(
      labels=tf.cast(y, dtype=tf.float32), logits=logits)
    loss = tf.reduce_mean(losses)

    optimizer = tf.train.AdamOptimizer(**adam_hparams).minimize(loss)

    tensors = utils.register_collection(
      hparams.collections,
      [("x", x),
       ("y", y),
       ("gamma", gamma),
       ("scores", scores),
       ("logits", logits),
       ("prob", prob),
       ("pred", pred),
       ("accu", accu),
       ("loss", loss),
       ("losses", losses),
       ("optimizer", optimizer),
      ])

    return tensors

  def train_step(self, sess, batch, gamma=1):
    loss, accu, _ = sess.run(
      [self.tensors["loss"], self.tensors["accu"],
       self.tensors["optimizer"]],
      feed_dict={
        context.is_train(): True,
        self.tensors["x"]: batch["x"],
        self.tensors["y"]: batch["y"],
        self.tensors["gamma"]: [gamma],
      })

    return loss, accu

  def eval_step(self, sess, batch, gamma=1):
    loss, prob = sess.run(
      # return loss as vector
      [self.tensors["losses"], self.tensors["prob"]],
      feed_dict={
        context.is_train(): False,
        self.tensors["x"]: batch["x"],
        self.tensors["y"]: batch["y"],
        self.tensors["gamma"]: [gamma],
      })

    return loss, prob

  def visualize_step(self, sess, batch, gamma=1):
    loss, prob, scores = sess.run(
      # return loss as vector
      [self.tensors["losses"], self.tensors["prob"], self.tensors["scores"]],
      feed_dict={
        context.is_train(): False,
        self.tensors["x"]: batch["x"],
        self.tensors["y"]: batch["y"],
        self.tensors["gamma"]: [gamma],
      })

    return loss, prob, scores 


