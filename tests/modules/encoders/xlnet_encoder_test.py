#
"""
Unit tests for XLNet encoders.
"""

import tensorflow as tf

from texar.tf.modules.encoders.xlnet_encoder import XLNetEncoder
from texar.tf.utils.test import pretrained_test


class XLNetEncoderTest(tf.test.TestCase):
    """Tests :class:`~texar.tf.modules.XLNetEncoder` class.
    """

    @pretrained_test
    def test_model_loading(self):
        r"""Tests model loading functionality."""

        inputs = tf.placeholder(dtype=tf.int32, shape=[None, None])

        for pretrained_model_name in XLNetEncoder.available_checkpoints():
            encoder = XLNetEncoder(pretrained_model_name=pretrained_model_name)
            _ = encoder(inputs)

    @pretrained_test
    def test_hparams(self):
        """Tests the priority of the encoder architecture parameter.
        """

        inputs = tf.placeholder(dtype=tf.int32, shape=[None, None])

        # case 1: set "pretrained_mode_name" by constructor argument
        encoder = XLNetEncoder(pretrained_model_name="xlnet-large-cased",
                               hparams={})
        encoder(inputs)
        self.assertEqual(len(encoder.attn_layers), 24)
        self.assertEqual(len(encoder.ff_layers), 24)

        # case 2: set "pretrained_mode_name" by hparams
        hparams = {
            "pretrained_model_name": "xlnet-base-cased"
        }
        encoder = XLNetEncoder(hparams=hparams)
        encoder(inputs)
        self.assertEqual(len(encoder.attn_layers), 12)
        self.assertEqual(len(encoder.ff_layers), 12)

        # case 3: set to None in both hparams and constructor argument
        # load no pre-trained model
        hparams = {
            "pretrained_model_name": None,
            "num_layers": 16
        }
        encoder = XLNetEncoder(hparams=hparams)
        encoder(inputs)
        self.assertEqual(len(encoder.attn_layers), 16)
        self.assertEqual(len(encoder.ff_layers), 16)

        # case 4: using default hparams
        encoder = XLNetEncoder()
        encoder(inputs)
        self.assertEqual(len(encoder.attn_layers), 12)
        self.assertEqual(len(encoder.ff_layers), 12)

    @pretrained_test
    def test_trainable_variables(self):
        """Tests the functionality of automatically collecting trainable
        variables.
        """

        inputs = tf.placeholder(dtype=tf.int32, shape=[None, None])

        # case 1: XLNet with no pre-trained model
        encoder = XLNetEncoder(hparams={
                                   "pretrained_model_name": None,
                                   "untie_r": False
                               })
        encoder(inputs)

        n_word_embed_vars = 1
        n_mask_embed_vars = 1
        n_bias_vars = 3  # r_r_bias, r_w_bias, r_s_bias
        n_pos_wise_ff_vars = 6  # 2 kernels + 2 bias + beta + gamma
        n_rel_multi_head_vars = 7  # q,k,v,r,o + beta + gamma
        n_segment_embed_vars = 1
        n_layers = encoder.hparams.num_layers
        n_trainable_variables = \
            n_word_embed_vars + n_segment_embed_vars + n_mask_embed_vars + \
            n_layers * (n_rel_multi_head_vars + n_pos_wise_ff_vars) + \
            n_bias_vars
        self.assertEqual(len(encoder.trainable_variables),
                         n_trainable_variables)

        # case 2: XLNet with pre-trained model
        hparams = {
            "pretrained_model_name": "xlnet-large-cased"
        }
        encoder = XLNetEncoder(hparams=hparams)
        encoder(inputs)
        n_segment_embed_vars = 1
        n_layers = encoder.hparams.num_layers
        n_trainable_variables = \
            n_word_embed_vars + n_segment_embed_vars + n_mask_embed_vars + \
            n_layers * (n_rel_multi_head_vars + n_pos_wise_ff_vars) \
            + n_bias_vars
        self.assertEqual(len(encoder.trainable_variables),
                         n_trainable_variables)

    def test_encode(self):
        """Tests encoding.
        """
        # case 1: XLNet pre-trained
        hparams = {
            "pretrained_model_name": None,
            "untie_r": False
        }
        encoder = XLNetEncoder(hparams=hparams)

        max_time = 8
        batch_size = 128
        inputs = tf.random_uniform([batch_size, max_time],
                                   maxval=30521, dtype=tf.int32)
        outputs, _ = encoder(inputs)

        outputs_dim = encoder.hparams.hidden_dim
        with self.session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs_ = sess.run(outputs)
            self.assertEqual(outputs_.shape,
                             (batch_size, max_time, outputs_dim))

        # case 2: XLNet pre-trained, untie_r=True
        hparams = {
            "pretrained_model_name": None,
            "untie_r": True
        }

        encoder = XLNetEncoder(hparams=hparams)
        outputs, _ = encoder(inputs)
        with self.session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs_ = sess.run(outputs)
            self.assertEqual(outputs_.shape,
                             (batch_size, max_time, outputs_dim))

        # case 3: XLNet with no pre-trained model
        hparams = {
            "pretrained_model_name": None
        }
        encoder = XLNetEncoder(hparams=hparams)
        outputs_dim = encoder.hparams.hidden_dim
        outputs, _ = encoder(inputs)
        with self.session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs_ = sess.run(outputs)
            self.assertEqual(outputs_.shape,
                             (batch_size, max_time, outputs_dim))


if __name__ == "__main__":
    tf.test.main()
