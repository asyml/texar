"""
Unit tests for GPT2 encoder.
"""

import tensorflow as tf

from texar.tf.modules.encoders.gpt2_encoder import GPT2Encoder
from texar.tf.utils.test import pretrained_test


class GPT2EncoderTest(tf.test.TestCase):
    r"""Tests :class:`~texar.torch.modules.GPT2Encoder` class.
    """

    @pretrained_test
    def test_model_loading(self):
        r"""Tests model loading functionality."""

        inputs = tf.placeholder(dtype=tf.int32, shape=[None, None])

        for pretrained_model_name in GPT2Encoder.available_checkpoints():
            encoder = GPT2Encoder(pretrained_model_name=pretrained_model_name)
            _ = encoder(inputs)

    @pretrained_test
    def test_hparams(self):
        """Tests the priority of the encoder arch parameter.
        """

        inputs = tf.placeholder(dtype=tf.int32, shape=[None, None])

        # case 1: set "pretrained_mode_name" by constructor argument
        hparams = {
            "pretrained_model_name": "gpt2-medium",
        }
        encoder = GPT2Encoder(pretrained_model_name="gpt2-small",
                              hparams=hparams)
        _ = encoder(inputs)
        self.assertEqual(encoder.hparams.encoder.num_blocks, 12)

        # case 2: set "pretrained_mode_name" by hparams
        hparams = {
            "pretrained_model_name": "gpt2-small",
            "encoder": {
                "num_blocks": 6
            }
        }
        encoder = GPT2Encoder(hparams=hparams)
        _ = encoder(inputs)
        self.assertEqual(encoder.hparams.encoder.num_blocks, 12)

        # case 3: set to None in both hparams and constructor argument
        hparams = {
            "pretrained_model_name": None,
            "encoder": {
                "num_blocks": 6
            },
        }
        encoder = GPT2Encoder(hparams=hparams)
        _ = encoder(inputs)
        self.assertEqual(encoder.hparams.encoder.num_blocks, 6)

        # case 4: using default hparams
        encoder = GPT2Encoder()
        _ = encoder(inputs)
        self.assertEqual(encoder.hparams.encoder.num_blocks, 12)

    @pretrained_test
    def test_trainable_variables(self):
        r"""Tests the functionality of automatically collecting trainable
        variables.
        """

        inputs = tf.placeholder(dtype=tf.int32, shape=[None, None])

        def get_variable_num(n_layers: int) -> int:
            return 1 + 1 + n_layers * 16 + 2

        # case 1: GPT2 small
        encoder = GPT2Encoder()
        _ = encoder(inputs)
        self.assertEqual(len(encoder.trainable_variables), get_variable_num(12))

        # case 2: GPT2 medium
        hparams = {
            "pretrained_model_name": "gpt2-medium",
        }
        encoder = GPT2Encoder(hparams=hparams)
        _ = encoder(inputs)
        self.assertEqual(len(encoder.trainable_variables), get_variable_num(24))

        # case 3: GPT2 large
        hparams = {
            "pretrained_model_name": "gpt2-large",
        }
        encoder = GPT2Encoder(hparams=hparams)
        _ = encoder(inputs)
        self.assertEqual(len(encoder.trainable_variables), get_variable_num(36))

        # case 4: self-designed GPT2
        hparams = {
            "pretrained_model_name": None,
            "encoder": {
                "num_blocks": 6
            },
        }
        encoder = GPT2Encoder(hparams=hparams)
        _ = encoder(inputs)
        self.assertEqual(len(encoder.trainable_variables), get_variable_num(6))

    def test_encode(self):
        r"""Tests encoding.
        """
        # case 1: GPT2 small
        hparams = {
            "pretrained_model_name": None
        }
        encoder = GPT2Encoder(hparams=hparams)

        max_time = 8
        batch_size = 16
        inputs = tf.random_uniform([batch_size, max_time],
                                   maxval=30521, dtype=tf.int32)
        outputs = encoder(inputs)

        outputs_dim = encoder.hparams.encoder.dim
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs_ = sess.run(outputs)
            self.assertEqual(outputs_.shape, (batch_size,
                                              max_time, outputs_dim))


if __name__ == "__main__":
    tf.test.main()
