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

from texar import context
from texar.hyperparams import HParams
from texar.core.utils import switch_dropout
from texar.modules.encoders.conv1d_discriminator import CNN
from texar.modules.decoders.rnn_decoders import BasicRNNDecoder
from texar.modules.decoders.rnn_decoder_helpers import *
from texar.models.tsf import ops
from texar.models.tsf import utils


class TSFClassifier:
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
      "embedding_size": 100,
      # rnn hprams
      "rnn_type": "GRUCell",
      "rnn_size": 700,
      "rnn_input_keep_prob": 0.5,
      "rnn_output_keep_prob": 0.5,
      "dim_y": 200,
      "dim_z": 500,
      # rnn decoder
      "decoder_use_embedding": False,
      "decoder_max_decoding_length_train": 21,
      "decoder_max_decoding_length_infer": 20,
      # cnn
      "cnn_name": "cnn",
      "cnn_use_embedding": True,
      "cnn_use_gate": False,
      "cnn_attn_size": 100,
      "cnn_kernel_sizes": [3, 4, 5],
      "cnn_num_filter": 128,
      "cnn_input_keep_prob": 1.,
      "cnn_output_keep_prob": 0.5,
      "cnn_vocab_size": 10000,
      "cnn_embedding_size": 100,
      # adam
      "adam_learning_rate": 1e-4,
      "adam_beta1": 0.9,
      "adam_beta2": 0.999
    }

  def _build_inputs(self):
    batch_size = self._hparams.batch_size

    enc_inputs = tf.placeholder(tf.int32, [batch_size, None], name="enc_inputs")
    dec_inputs = tf.placeholder(tf.int32, [batch_size, None], name="dec_inputs")
    targets = tf.placeholder(tf.int32, [batch_size, None], name="targets")
    weights = tf.placeholder(tf.float32, [batch_size, None], name="weights")
    labels = tf.placeholder(tf.float32, [batch_size], name="labels")
    seq_len = tf.placeholder(tf.int32, [batch_size], name="seq_len")
    gamma = tf.placeholder(tf.float32, [], name="gamma")
    rho_f = tf.placeholder(tf.float32, [], name="rho_f")
    rho_r = tf.placeholder(tf.float32, [], name="rho_r")

    collections_input = self._hparams.collections + '/input'
    input_tensors = utils.register_collection(
      collections_input,
      [("enc_inputs", enc_inputs),
       ("dec_inputs", dec_inputs),
       ("targets", targets),
       ("weights", weights),
       ("labels", labels),
       ("seq_len", seq_len),
       ("gamma", gamma),
       ("rho_f", rho_f),
       ("rho_r", rho_r),
      ])

    return input_tensors

  def _build_model(self, input_tensors, reuse=False):
    hparams = self._hparams
    embedding_init = np.random.random_sample(
      (hparams.vocab_size, hparams.embedding_size)) - 0.5
    for i in range(hparams.vocab_size):
      embedding_init[i] /= np.linalg.norm(embedding_init[i])
    # embedding = tf.get_variable(
    #   "embedding", shape=[hparams.vocab_size, hparams.embedding_size])
    embedding = tf.get_variable(
      "embedding", initializer=embedding_init.astype(np.float32))

    enc_inputs = tf.nn.embedding_lookup(embedding, input_tensors["enc_inputs"])
    dec_inputs = tf.nn.embedding_lookup(embedding, input_tensors["dec_inputs"])

    labels = input_tensors["labels"]
    labels = tf.reshape(labels, [-1, 1])

    rnn_hparams = utils.filter_hparams(hparams, "rnn")
    init_state = tf.zeros([hparams.batch_size, rnn_hparams.size])
    cell_e = ops.get_rnn_cell(rnn_hparams)

    _, z = tf.nn.dynamic_rnn(cell_e, enc_inputs, initial_state=init_state,
                             scope="encoder")
    z  = z[:, hparams.dim_y:]

    label_proj_g = tf.layers.Dense(hparams.dim_y, name="generator")
    h_ori = tf.concat([label_proj_g(labels), z], 1)
    h_tsf = tf.concat([label_proj_g(1 - labels), z], 1)

    cell_g = ops.get_rnn_cell(rnn_hparams)
    softmax_proj = tf.layers.Dense(hparams.vocab_size, name="softmax_proj")
    g_outputs, _ = tf.nn.dynamic_rnn(cell_g, dec_inputs, initial_state=h_ori,
                                     scope="generator")

    g_logits = softmax_proj(tf.reshape(g_outputs, [-1, rnn_hparams.size]))

    loss_g = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=tf.reshape(input_tensors["targets"], [-1]), logits=g_logits)
    loss_g *= tf.reshape(input_tensors["weights"], [-1])
    ppl_g = tf.reduce_sum(loss_g) / (tf.reduce_sum(input_tensors["weights"]) \
                                     + 1e-8)
    loss_g = tf.reduce_sum(loss_g) / hparams.batch_size

    # classifier supervised training 
    half = hparams.batch_size // 2
    cnn_hparams = utils.filter_hparams(hparams, "cnn")
    cnn = CNN(cnn_hparams)
    targets = input_tensors["targets"]
    loss_ds, accu_s, mask1, mask0 \
      = ops.adv_loss(targets[half:],
                     targets[:half],
                     cnn,
                     input_tensors["seq_len"][half:],
                     input_tensors["seq_len"][:half])
    mask = tf.concat([mask0, mask1], axis=0)

    # decoding 
    decoder_hparams = utils.filter_hparams(hparams, "decoder")
    decoder = BasicRNNDecoder(cell=cell_g, output_layer=softmax_proj,
                              hparams=decoder_hparams)
    start_tokens = input_tensors["dec_inputs"][:, 0]
    start_tokens = tf.reshape(start_tokens, [-1])
    #TODO(zichao): hard coded end_token
    end_token = 2
    gumbel_helper = GumbelSoftmaxEmbeddingHelper(
      embedding, start_tokens, end_token, input_tensors["gamma"])
    greedy_helper = GreedyEmbeddingHelper(embedding, start_tokens, end_token)

    soft_outputs_ori, _, soft_len_ori = decoder(gumbel_helper, h_ori)
    soft_outputs_tsf, _, soft_len_tsf = decoder(gumbel_helper, h_tsf)

    hard_outputs_ori, _, hard_len_ori = decoder(greedy_helper, h_ori)
    hard_outputs_tsf, _, hard_len_tsf = decoder(greedy_helper, h_tsf)

    # discriminator

    loss_dr, accu_r, _, _ = ops.adv_loss(soft_sample_ori.predicted_ids[half:],
                                         soft_sample_ori.predicted_ids[:half],
                                         cnn,
                                         soft_len_ori[half:],
                                         soft_len_ori[:half])
    loss_df, accu_f, _, _ = ops.adv_loss(soft_sample_tsf.predicted_ids[:half],
                                         soft_sample_tsf.predicted_ids[half:],
                                         cnn,
                                         soft_sample_tsf[:half],
                                         soft_sample_tsf[half:])

    loss = loss_g + \
           input_tensors["rho_f"] * loss_df + \
           input_tensors["rho_r"] * loss_dr

    var_eg = ops.retrieve_variables(["encoder", "generator", "softmax_proj",
                                     "embedding"])
    var_eg += decoder.trainavle_variables
    var_d = ops.retrieve_variables(["cnn"])

    # optimization
    adam_hparams = utils.filter_hparams(hparams, "adam")
    optimizer_all = tf.train.AdamOptimizer(**adam_hparams).minimize(
      loss, var_list=var_eg)
    optimizer_ae = tf.train.AdamOptimizer(**adam_hparams).minimize(
      loss_g, var_list=var_eg)
    optimizer_ds = tf.train.AdamOptimizer(**adam_hparams).minimize(
      loss_ds, var_list=var_d)

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
       ("g_logits", g_logits),
       ("mask", mask),
       ("soft_len_ori", soft_len_ori),
       ("soft_len_tsf", soft_len_tsf),
      ]
    )

    collections_loss = hparams.collections + '/loss'
    loss = utils.register_collection(
      collections_loss,
      [("loss", loss),
       ("loss_g", loss_g),
       ("ppl_g", ppl_g),
       ("loss_ds", loss_ds),
       ("loss_df", loss_df),
       ("loss_dr", loss_dr),
       ("accu_s", accu_s),
       ("accu_f", accu_f),
       ("accu_r", accu_r),
      ]
    )

    collections_opt = hparams.collections + '/opt'
    opt = utils.register_collection(
      collections_opt,
      [("optimizer_all", optimizer_all),
       ("optimizer_ae", optimizer_ae),
       ("optimizer_ds", optimizer_ds),
      ]
    )

    return output_tensors, loss, opt

  def train_d_step(self, sess, batch):
    loss_ds, accu_s, _ = sess.run(
      [self.loss["loss_ds"],
       self.loss["accu_s"],
       self.opt["optimizer_ds"]],
      self.feed_dict(batch, 0., 0., 1.))
    return loss_ds, accu_s

  def train_g_step(self, sess, batch, rho_f, rho_r, gamma):
    loss, loss_g, ppl_g, loss_df, loss_dr, accu_f, accu_r, _ = sess.run(
      [self.loss["loss"],
       self.loss["loss_g"],
       self.loss["ppl_g"],
       self.loss["loss_df"],
       self.loss["loss_dr"],
       self.loss["accu_f"],
       self.loss["accu_r"],
       self.opt["optimizer_all"]],
      self.feed_dict(batch, rho_f, rho_r, gamma))
    return loss, loss_g, ppl_g, loss_df, loss_dr, accu_f, accu_r

  def train_ae_step(self, sess, batch, rho_f, rho_r, gamma):
    loss, loss_g, ppl_g, loss_df, loss_dr, accu_f, accu_r, _ = sess.run(
      [self.loss["loss"],
       self.loss["loss_g"],
       self.loss["ppl_g"],
       self.loss["loss_df"],
       self.loss["loss_dr"],
       self.loss["accu_f"],
       self.loss["accu_r"],
       self.opt["optimizer_ae"]],
      self.feed_dict(batch, rho_f, rho_r, gamma))
    return loss, loss_g, ppl_g, loss_df, loss_dr, accu_f, accu_r

  def eval_step(self, sess, batch, rho_f, rho_r, gamma):
    loss, loss_g, ppl_g, loss_df, loss_dr, loss_ds, \
      accu_f, accu_r, accu_s = sess.run(
      [self.loss["loss"],
       self.loss["loss_g"],
       self.loss["ppl_g"],
       self.loss["loss_df"],
       self.loss["loss_dr"],
       self.loss["loss_ds"],
       self.loss["accu_f"],
       self.loss["accu_r"],
       self.loss["accu_s"],
      ],
      self.feed_dict(batch, rho_f, rho_r, gamma, is_train=False))
    return (loss, loss_g, ppl_g, loss_df, loss_dr, loss_ds,
            accu_f, accu_r, accu_s)

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

  def feed_dict(self, batch, rho_f, rho_r, gamma, is_train=True):
    return {
      context.is_train(): is_train,
      self.input_tensors["seq_len"]: batch["seq_len"],
      self.input_tensors["enc_inputs"]: batch["enc_inputs"],
      self.input_tensors["dec_inputs"]: batch["dec_inputs"],
      self.input_tensors["targets"]: batch["targets"],
      self.input_tensors["weights"]: batch["weights"],
      self.input_tensors["labels"]: batch["labels"],
      self.input_tensors["rho_f"]: rho_f,
      self.input_tensors["rho_r"]: rho_r,
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
