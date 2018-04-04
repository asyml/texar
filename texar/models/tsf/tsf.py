"""
Text Style Transfer.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb

import copy
import tensorflow as tf
from tensorflow.contrib.seq2seq import GreedyEmbeddingHelper

import texar as tx
from texar import context
from texar.utils import switch_dropout
from texar.modules.embedders import WordEmbedder
# from texar.modules.classifiers import Conv1DClassifier
from texar.modules.classifiers import CNN as Conv1DClassifier
from texar.modules.encoders import UnidirectionalRNNEncoder
from texar.modules.decoders import BasicRNNDecoder
from texar.modules.decoders import _get_training_helper, GumbelSoftmaxEmbeddingHelper
from texar.modules.connectors import MLPTransformConnector
from texar.core.layers import SequentialLayer
from texar.core import get_train_op
from texar.losses import adv_losses
from texar.models import ModelBase


class TSF(ModelBase):
    """Text style transfer."""

    def __init__(self, hparams=None):
        ModelBase.__init__(self, hparams)
        self.build_inputs()
        self.build_model(self.input_tensors)
        self.saver = tf.train.Saver()

    @staticmethod
    def default_hparams():
        return {
            "name": "tsf",
            "collections": "tsf",
            "vocab_size": 10000,
            "batch_size": 128,
            "output_keep_prob": 0.5,
            "dim_y": 200,
            "dim_z": 500,
            "embedder": {
                "dim": 100,
            },
            "rnn_encoder": {
                "rnn_cell": {
                    "type": "GRUCell",
                    "kwargs": {
                        "num_units": 700
                    },
                    "dropout": {
                        "input_keep_prob": 0.5
                    }
                }
            },
            "rnn_decoder": {
                "rnn_cell": {
                    "type": "GRUCell",
                    "kwargs": {
                        "num_units": 700,
                    },
                    "dropout": {
                        "input_keep_prob": 0.5,
                    },
                },
                "max_decoding_length_train": 21,
                "max_decoding_length_infer": 20,
            },
            "cnn": {
                "name": "cnn",
                "use_embedding": False,
                "vocab_size": 10000,
                "kernel_sizes": [3, 4, 5],
                "num_filter": 128,
                "output_keep_prob": 0.5,
                "input_keep_prob": 1.,
            },
            "opt": {
                "name": "optimizer",
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

    def build_inputs(self):
        batch_size = self._hparams.batch_size
        enc_inputs = tf.placeholder(tf.int32, [batch_size, None],
                                    name="enc_inputs")
        dec_inputs = tf.placeholder(tf.int32, [batch_size, None],
                                    name="dec_inputs")
        targets = tf.placeholder(tf.int32, [batch_size, None],
                                 name="targets")
        seq_len = tf.placeholder(tf.int32, [batch_size], name="seq_len")
        labels = tf.placeholder(tf.float32, [batch_size], name="labels")
        gamma = tf.placeholder(tf.float32, [], name="gamma")

        self.input_tensors = {
            "enc_inputs": enc_inputs,
            "dec_inputs": dec_inputs,
            "targets": targets,
            "seq_len": seq_len,
            "labels": labels,
            "gamma": gamma,
        }

    def build_model(self, input_tensors):
        hparams = self._hparams
        
        labels = tf.reshape(input_tensors["labels"], [-1, 1])
      
        embedder = WordEmbedder(vocab_size=hparams.vocab_size)
        # encoder
        rnn_encoder = UnidirectionalRNNEncoder(hparams=hparams.rnn_encoder)
        # seq_len - 1 to remove EOS
        enc_inputs = embedder(input_tensors["enc_inputs"])
        _, z = rnn_encoder(enc_inputs,
                           sequence_length=input_tensors["seq_len"]-1)
        z = z[:, hparams.dim_y:]

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
        train_helper = _get_training_helper(input_tensors["dec_inputs"], seq_len,
                                            embedding=embedder)
        g_outputs, _, _ = rnn_decoder(helper=train_helper, initial_state=h_ori)

        teach_h = tf.concat([tf.expand_dims(h_ori, 1), g_outputs.cell_output], 1)

        loss_g = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=input_tensors["targets"], logits=g_outputs.logits)
        mask = tf.sequence_mask(input_tensors["seq_len"], tf.shape(loss_g)[1],
                                tf.float32)
        loss_g *= mask
        ppl_g = tf.reduce_sum(loss_g) / (tf.reduce_sum(mask) + 1e-8)
        loss_g = tf.reduce_sum(loss_g) / hparams.batch_size

        # gumbel and greedy decoder
        start_tokens = input_tensors["dec_inputs"][:, 0]
        start_tokens = tf.reshape(start_tokens, [-1])
        end_token = 2
        gumbel_helper = GumbelSoftmaxEmbeddingHelper(
            embedder.embedding, start_tokens, end_token, input_tensors["gamma"],
            use_finish=False)

        #TODO(zichao): hard coded end_token
        end_token = 2
        greedy_helper = GreedyEmbeddingHelper(
            embedder.embedding, start_tokens, end_token)

        soft_outputs_ori, _, _, = rnn_decoder(
            helper=gumbel_helper, initial_state=h_ori)
        soft_outputs_tsf, _, _, = rnn_decoder(
            helper=gumbel_helper, initial_state=h_tsf)

        hard_outputs_ori, _, _, = rnn_decoder(
            helper=greedy_helper, initial_state=h_ori)
        hard_outputs_tsf, _, _, = rnn_decoder(
            helper=greedy_helper, initial_state=h_tsf)

        # classifier
        half = hparams.batch_size // 2
        h_len = tf.shape(g_outputs.cell_output)[1]

        cnn_hparams = hparams.cnn
        cnn_hparams.use_embedding = True
        cnn_hparams.vocab_size = hparmas.vocab_size

        targets = input_tensors["targets"]
        tsf_sample_id = soft_outputs_tsf.sample_id
        tsf_sample_id = tsf_sample_id[:, :h_len, :]
        cnn = Conv1DClassifier(hparams.cnn)
        _, loss_ds = adv_losses.binary_adversarial_losses(
            targets[half:], targets[:half], cnn)
        _, loss_df = adv_losses.binary_adversarial_losses(
            tsf_sample_id[:half], tsf_sample_id[half:], cnn)

        # discriminator
        # plus the encoder h
        soft_output_tsf \
            = soft_outputs_tsf.cell_output[:, :h_len, :]
        soft_h_tsf = tf.concat([tf.expand_dims(h_tsf, 1), soft_output_tsf], 1)

        cnn0_hparams = copy.deepcopy(hparams.cnn)
        cnn1_hparams = copy.deepcopy(hparams.cnn)
        cnn0_hparams.name = "cnn0"
        cnn1_hparams.name = "cnn1"
    
        cnn0 = Conv1DClassifier(cnn0_hparams)
        cnn1 = Conv1DClassifier(cnn1_hparams)

        _, loss_d0 = adv_losses.binary_adversarial_losses(
            teach_h[:half], soft_h_tsf[half:], cnn0)
        _, loss_d1 = adv_losses.binary_adversarial_losses(
            teach_h[half:], soft_h_tsf[:half], cnn1)

        loss_d = loss_d0 + loss_d1
        loss = loss_g
        if hparams.rho_adv > 0.:
            loss -= hparmas.rho_adv * loss_d
        if hparams.rho_f > 0:
            loss += hparams.rho_f * loss_df

        var_eg = embedder.trainable_variables + \
                 rnn_encoder.trainable_variables + \
                 rnn_decoder.trainable_variables \
                 + label_proj_g.trainable_variables
        var_ds = cnn.trainable_variables
        var_d0 = cnn0.trainable_variables
        var_d1 = cnn1.trainable_variables

        # optimization
        opt_all_hparams = copy.deepcopy(hparams.opt)
        opt_ae_hparams = copy.deepcopy(hparams.opt)
        opt_ds_hparams = copy.deepcopy(hparams.opt)
        opt_d0_hparams = copy.deepcopy(hparams.opt)
        opt_d1_hparams = copy.deepcopy(hparams.opt)
        opt_all_hparams.name = "optimizer_all"
        opt_ae_hparams.name = "optimizer_ae"
        opt_ds_hparams.name = "optimizer_ds"
        opt_d0_hparams.name = "optimizer_d0"
        opt_d1_hparams.name = "optimizer_d1"
        optimizer_all = get_train_op(loss, variables=var_eg,
                                     hparams=opt_all_hparams)
        optimizer_ae = get_train_op(loss_g, variables=var_eg,
                                    hparams=opt_ae_hparams)
        optimizer_ds = get_train_op(loss_ds, variables=var_ds,
                                    hparams=opt_ds_hparams)
        optimizer_d0 = get_train_op(loss_d0, variables=var_d0,
                                    hparams=opt_d0_hparams)
        optimizer_d1 = get_train_op(loss_d1, variables=var_d1,
                                    hparams=opt_d1_hparams)
        
        self.output_tensors = {
            "h_ori": h_ori,
            "h_tsf": h_tsf,
            "hard_logits_ori": hard_outputs_ori.logits,
            "hard_logits_tsf": hard_outputs_tsf.logits,
            "soft_logits_ori": soft_outputs_ori.logits,
            "soft_logits_tsf": soft_outputs_tsf.logits,
            "g_logits": g_outputs.logits,
            "teach_h": teach_h,
            "soft_h_tsf": soft_h_tsf,
        }

        self.loss = {
            "loss": loss,
            "loss_g": loss_g,
            "ppl_g": ppl_g,
            "loss_d": loss_d,
            "loss_d0": loss_d0,
            "loss_d1": loss_d1,
            "loss_ds": loss_ds,
            "loss_df": loss_df
        }

        self.opt = {
            "optimizer_all": optimizer_all,
            "optimizer_ae": optimizer_ae,
            "optimizer_ds": optimizer_ds,
            "optimizer_d0": optimizer_d0,
            "optimizer_d1": optimizer_d1,
        }

    def train_ds_step(self, sess, batch, gamma):
        loss_ds, _ = sess.run(
            [self.loss["loss_ds"], self.opt["optimizer_ds"],],
            self.feed_dict(batch, gamma))
        return loss_ds

    def train_d0_step(self, sess, batch,  gamma):
        loss_d0, _ = sess.run(
            [self.loss["loss_d0"], self.opt["optimizer_d0"],],
            self.feed_dict(batch, gamma))
        return loss_d0

    def train_d1_step(self, sess, batch, gamma):
        loss_d1, _ = sess.run(
            [self.loss["loss_d1"], self.opt["optimizer_d1"]],
            self.feed_dict(batch, gamma))
        return loss_d1

    def train_g_step(self, sess, batch, gamma):
        loss, loss_g, ppl_g, loss_d, loss_df, _ = sess.run(
            [self.loss["loss"],
             self.loss["loss_g"],
             self.loss["ppl_g"],
             self.loss["loss_d"],
             self.loss["loss_df"],
             self.opt["optimizer_all"]],
            self.feed_dict(batch, gamma))
        return loss, loss_g, ppl_g, loss_d, loss_df

    def train_ae_step(self, sess, batch, gamma):
        loss, loss_g, ppl_g, loss_d, loss_df, _ = sess.run(
            [self.loss["loss"],
             self.loss["loss_g"],
             self.loss["ppl_g"],
             self.loss["loss_d"],
             self.loss["loss_df"],
             self.opt["optimizer_ae"]],
            self.feed_dict(batch, gamma))
        return loss, loss_g, ppl_g, loss_d, loss_df

    def eval_step(self, sess, batch, gamma):
        loss, loss_g, ppl_g, loss_d, loss_d0, loss_d1, \
            loss_ds, loss_df = sess.run(
            [self.loss["loss"],
             self.loss["loss_g"],
             self.loss["ppl_g"],
             self.loss["loss_d"],
             self.loss["loss_d0"],
             self.loss["loss_d1"],
             self.loss["loss_ds"],
             self.loss["loss_df"],],
            self.feed_dict(batch, gamma,
                           mode=tf.estimator.ModeKeys.EVAL))
        return (loss, loss_g, ppl_g, loss_d, loss_d0, loss_d1,
                loss_ds, loss_df)

    def decode_step(self, sess, batch):
        logits_ori, logits_tsf = sess.run(
            [self.output_tensors["hard_logits_ori"],
             self.output_tensors["hard_logits_tsf"]],
            feed_dict={
                tx.global_mode(): tf.estimator.ModeKeys.EVAL,
                self.input_tensors["enc_inputs"]: batch["enc_inputs"],
                self.input_tensors["dec_inputs"]: batch["dec_inputs"],
                self.input_tensors["labels"]: batch["labels"],
                self.input_tensors["seq_len"]: batch["seq_len"],
            })
        return logits_ori, logits_tsf

    def feed_dict(self, batch, gamma, mode=tf.estimator.ModeKeys.TRAIN):
        return {
            tx.global_mode(): mode,
            self.input_tensors["enc_inputs"]: batch["enc_inputs"],
            self.input_tensors["dec_inputs"]: batch["dec_inputs"],
            self.input_tensors["targets"]: batch["targets"],
            self.input_tensors["seq_len"]: batch["seq_len"],
            self.input_tensors["labels"]: batch["labels"],
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
