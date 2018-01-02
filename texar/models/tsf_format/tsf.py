"""
Text Style Transfer.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb

import copy
import numpy as np
import tensorflow as tf
from tensorflow.contrib.seq2seq import GreedyEmbeddingHelper

from texar import context
from texar.hyperparams import HParams
from texar.core.utils import switch_dropout
from texar.modules.encoders.conv1d_discriminator import CNN
from texar.modules.encoders.rnn_encoders import UnidirectionalRNNEncoder
from texar.modules.decoders.rnn_decoders import BasicRNNDecoder
from texar.modules.decoders.rnn_decoder_helpers import *
from texar.modules.connectors import MLPTransformConnector
from texar.core.layers import *
from texar.core import optimization
from texar.losses import adv_losses
from texar.models.tsf_format import utils


class TSF:
  """Text style transfer."""

  def __init__(self, hparams=None):
    self._hparams = HParams(hparams, self.default_hparams(),
                            allow_new_hparam=True)
    self.input_tensors = self._build_inputs()
    (self.output_tensors, self.loss, self.opt) \
      = self._build_model(self.input_tensors)
    self.saver = tf.train.Saver()

  @staticmethod
  def default_hparams():
    return {
      "name": "text_style_transfer",
      "collections": "tsf",
      "batch_size": 128,
      "rnn_encoder": {
        "embedding": {
          "dim": 100,
        },
        "rnn_cell": {
          "cell": {
            "type": "GRUCell",
            "kwargs": {
              "num_units": 700
            },
          },
          "dropout": {
            "input_keep_prob": 0.5
          }
        }
      },
      "rnn_decoder": {
        "rnn_cell": {
          "cell": {
            "type": "GRUCell",
            "kwargs": {
              "num_units": 700,
            },
          },
          "dropout": {
            "input_keep_prob": 0.5,
          },
        },
        "use_embedding": False,
        "max_decoding_length_train": 21,
        "max_decoding_length_infer": 20,
      },
      "output_keep_prob": 0.5,
      "dim_y": 200,
      "dim_z": 500,
      "cnn": {
        "name": "cnn",
        "kernel_sizes": [3, 4, 5],
        "num_filter": 128,
        "input_keep_prob": 1.,
        "output_keep_prob": 0.5,
      },
      "opt": {
        "name": "opt",
        "optimizer": {
          "type":  "AdamOptimizer",
          "kwargs": {
            "learning_rate": 1e-4,
            "beta1": 0.9,
            "beta2": 0.999,
          },
        },
      },
    }

  def _build_inputs(self):
    batch_size = self._hparams.batch_size

    enc_inputs = tf.placeholder(tf.int32, [batch_size, None], name="enc_inputs")
    dec_inputs = tf.placeholder(tf.int32, [batch_size, None], name="dec_inputs")
    targets = tf.placeholder(tf.int32, [batch_size, None], name="targets")
    weights = tf.placeholder(tf.float32, [batch_size, None], name="weights")
    labels = tf.placeholder(tf.float32, [batch_size], name="labels")
    batch_len = tf.placeholder(tf.int32, name="batch_len")
    gamma = tf.placeholder(tf.float32, [], name="gamma")
    rho = tf.placeholder(tf.float32, [], name="rho")

    collections_input = self._hparams.collections + '/input'
    input_tensors = utils.register_collection(
      collections_input,
      [("enc_inputs", enc_inputs),
       ("dec_inputs", dec_inputs),
       ("targets", targets),
       ("weights", weights),
       ("labels", labels),
       ("batch_len", batch_len),
       ("gamma", gamma),
       ("rho", rho),
      ])

    return input_tensors

  def _build_model(self, input_tensors, reuse=False):
    hparams = self._hparams

    labels = tf.reshape(input_tensors["labels"], [-1, 1])

    # encoder
    rnn_encoder = UnidirectionalRNNEncoder(vocab_size=hparams.vocab_size,
                                           hparams=hparams.rnn_encoder)
    _, z = rnn_encoder(input_tensors["enc_inputs"])
    z  = z[:, hparams.dim_y:]

    # get state
    label_proj_g = MLPTransformConnector(hparams.dim_y)
    h_ori = tf.concat([label_proj_g(labels), z], 1)
    h_tsf = tf.concat([label_proj_g(1-labels), z], 1)

    output_dropout = tf.layers.Dropout(
      rate=1-switch_dropout(hparams.output_keep_prob))
    softmax_proj = tf.layers.Dense(hparams.vocab_size, name="softmax_proj")
    output_layer = SequentialLayer([output_dropout, softmax_proj])
    rnn_decoder = BasicRNNDecoder(output_layer=output_layer,
                                  hparams=hparams.rnn_decoder)

    seq_len = [tf.shape(input_tensors["dec_inputs"])[1]] * hparams.batch_size
    train_helper = EmbeddingTrainingHelper(input_tensors["dec_inputs"],
                                           seq_len,
                                           rnn_encoder.embedding)
    g_outputs, _, _ = rnn_decoder(train_helper, h_ori)

    teach_h = tf.concat([tf.expand_dims(h_ori, 1), g_outputs.cell_output], 1)

    loss_g = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=input_tensors["targets"], logits=g_outputs.logits)
    loss_g *= input_tensors["weights"]
    ppl_g = tf.reduce_sum(loss_g) / (tf.reduce_sum(input_tensors["weights"]) \
                                     + 1e-8)
    loss_g = tf.reduce_sum(loss_g) / hparams.batch_size

    # gumbel and greedy decoder
    start_tokens = input_tensors["dec_inputs"][:, 0]
    start_tokens = tf.reshape(start_tokens, [-1])
    gumbel_helper = GumbelSoftmaxEmbeddingHelper(
      rnn_encoder.embedding,
      start_tokens,
      input_tensors["gamma"],
    )
    # softmax_helper = SoftmaxEmbeddingHelper(
    #   rnn_encoder.embedding,
    #   start_tokens,
    #   input_tensors["gamma"],
    # )


    #TODO(zichao): hard coded end_token
    end_token = 2
    greedy_helper = GreedyEmbeddingHelper(
      rnn_encoder.embedding,
      start_tokens,
      end_token,
    )

    soft_outputs_ori, _, _, = rnn_decoder(gumbel_helper, h_ori)
    soft_outputs_tsf, _, _, = rnn_decoder(gumbel_helper, h_tsf)

    # soft_outputs_ori, _, _, = rnn_decoder(softmax_helper, h_ori)
    # soft_outputs_tsf, _, _, = rnn_decoder(softmax_helper, h_tsf)

    hard_outputs_ori, _, _, = rnn_decoder(greedy_helper, h_ori)
    hard_outputs_tsf, _, _, = rnn_decoder(greedy_helper, h_tsf)

    # discriminator
    half = hparams.batch_size // 2
    # plus the encoder h
    soft_output_tsf \
      = soft_outputs_tsf.cell_output[:, :input_tensors["batch_len"], :]
    soft_h_tsf = tf.concat([tf.expand_dims(h_tsf, 1), soft_output_tsf], 1)

    cnn0_hparams = copy.deepcopy(hparams.cnn)
    cnn1_hparams = copy.deepcopy(hparams.cnn)
    cnn0_hparams.name = "cnn0"
    cnn1_hparams.name = "cnn1"
    
    cnn0 = CNN(cnn0_hparams)
    cnn1 = CNN(cnn1_hparams)

    _, loss_d0 = adv_losses.binary_adversarial_losses(
      teach_h[:half], soft_h_tsf[half:], cnn0)
    _, loss_d1 = adv_losses.binary_adversarial_losses(
      teach_h[half:], soft_h_tsf[:half], cnn1)

    loss_d = loss_d0 + loss_d1
    loss = loss_g - input_tensors["rho"] * loss_d

    var_eg = rnn_encoder.trainable_variables + rnn_decoder.trainable_variables \
             + label_proj_g.trainable_variables
    var_d0 = cnn0.trainable_variables
    var_d1 = cnn1.trainable_variables

    # optimization
    opt_all_hparams = copy.deepcopy(hparams.opt)
    opt_ae_hparams = copy.deepcopy(hparams.opt)
    opt_d0_hparams = copy.deepcopy(hparams.opt)
    opt_d1_hparams = copy.deepcopy(hparams.opt)
    opt_all_hparams.name = "optimizer_all"
    opt_ae_hparams.name = "optimizer_ae"
    opt_d0_hparams.name = "optimizer_d0"
    opt_d1_hparams.name = "optimizer_d1"
    optimizer_all, _ = optimization.get_train_op(loss, variables=var_eg,
                                                 hparams=opt_all_hparams)
    optimizer_ae, _ = optimization.get_train_op(loss_g, variables=var_eg,
                                                hparams=opt_ae_hparams)
    optimizer_d0, _ = optimization.get_train_op(loss_d0, variables=var_d0,
                                                hparams=opt_d0_hparams)
    optimizer_d1, _ = optimization.get_train_op(loss_d1, variables=var_d1,
                                                hparams=opt_d1_hparams)

    # add tensors to collections
    collections_output = hparams.collections + '/output'
    output_tensors = utils.register_collection(
      collections_output,
      [("h_ori", h_ori),
       ("h_tsf", h_tsf),
       ("hard_logits_ori", hard_outputs_ori.logits),
       ("hard_logits_tsf", hard_outputs_tsf.logits),
       ("soft_logits_ori", soft_outputs_ori.logits),
       ("soft_logits_tsf", soft_outputs_tsf.logits),
       ("g_logits", g_outputs.logits),
       ("teach_h", teach_h),
       ("soft_h_tsf", soft_h_tsf),
      ]
    )

    collections_loss = hparams.collections + '/loss'
    loss = utils.register_collection(
      collections_loss,
      [("loss", loss),
       ("loss_g", loss_g),
       ("ppl_g", ppl_g),
       ("loss_d", loss_d),
       ("loss_d0", loss_d0),
       ("loss_d1", loss_d1),
      ]
    )

    collections_opt = hparams.collections + '/opt'
    opt = utils.register_collection(
      collections_opt,
      [("optimizer_all", optimizer_all),
       ("optimizer_ae", optimizer_ae),
       ("optimizer_d0", optimizer_d0),
       ("optimizer_d1", optimizer_d1),
      ]
    )

    return output_tensors, loss, opt

  def train_d0_step(self, sess, batch, rho, gamma):
    loss_d0, _ = sess.run(
      [self.loss["loss_d0"], self.opt["optimizer_d0"],],
      self.feed_dict(batch, rho, gamma))
    return loss_d0

  def train_d1_step(self, sess, batch, rho, gamma):
    loss_d1, _ = sess.run([self.loss["loss_d1"], self.opt["optimizer_d1"]],
                          self.feed_dict(batch, rho, gamma))
    return loss_d1

  def train_g_step(self, sess, batch, rho, gamma):
    loss, loss_g, ppl_g, loss_d, _ = sess.run(
      [self.loss["loss"],
       self.loss["loss_g"],
       self.loss["ppl_g"],
       self.loss["loss_d"],
       self.opt["optimizer_all"]],
      self.feed_dict(batch, rho, gamma))
    return loss, loss_g, ppl_g, loss_d

  def train_ae_step(self, sess, batch, rho, gamma):
    loss, loss_g, ppl_g, loss_d, _ = sess.run(
      [self.loss["loss"],
       self.loss["loss_g"],
       self.loss["ppl_g"],
       self.loss["loss_d"],
       self.opt["optimizer_ae"]],
      self.feed_dict(batch, rho, gamma))
    return loss, loss_g, ppl_g, loss_d

  def eval_step(self, sess, batch, rho, gamma):
    loss, loss_g, ppl_g, loss_d, loss_d0, loss_d1 = sess.run(
      [self.loss["loss"],
       self.loss["loss_g"],
       self.loss["ppl_g"],
       self.loss["loss_d"],
       self.loss["loss_d0"],
       self.loss["loss_d1"]],
      self.feed_dict(batch, rho, gamma, is_train=False))
    return loss, loss_g, ppl_g, loss_d, loss_d0, loss_d1

  def decode_step(self, sess, batch):
    logits_ori, logits_tsf = sess.run(
      [self.output_tensors["hard_logits_ori"],
       self.output_tensors["hard_logits_tsf"]],
      feed_dict={
        context.is_train(): False,
        self.input_tensors["enc_inputs"]: batch["enc_inputs"],
        self.input_tensors["dec_inputs"]: batch["dec_inputs"],
        self.input_tensors["labels"]: batch["labels"]})
    return logits_ori, logits_tsf

  def feed_dict(self, batch, rho, gamma, is_train=True):
    return {
      context.is_train(): is_train,
      self.input_tensors["batch_len"]: batch["len"],
      self.input_tensors["enc_inputs"]: batch["enc_inputs"],
      self.input_tensors["dec_inputs"]: batch["dec_inputs"],
      self.input_tensors["targets"]: batch["targets"],
      self.input_tensors["weights"]: batch["weights"],
      self.input_tensors["labels"]: batch["labels"],
      self.input_tensors["rho"]: rho,
      self.input_tensors["gamma"]: gamma,
    }

  def decode_step_soft(self, sess, batch, gamma=0.01):
    logits_ori, logits_tsf, g_logits, test_output, test_logits = sess.run(
      [self.output_tensors["soft_logits_ori"],
       self.output_tensors["soft_logits_tsf"],
       self.output_tensors["g_logits"],
       self.output_tensors["test_output"],
       self.output_tensors["test_logits"]],
      feed_dict={
        context.is_train(): False,
        self.input_tensors["enc_inputs"]: batch["enc_inputs"],
        self.input_tensors["dec_inputs"]: batch["dec_inputs"],
        self.input_tensors["labels"]: batch["labels"],
        self.input_tensors["gamma"]: gamma})
    return logits_ori, logits_tsf, g_logits, test_output, test_logits
