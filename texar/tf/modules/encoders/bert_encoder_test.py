"""
Unit tests for BERT encoders.
"""

import tensorflow as tf

from texar.tf.modules.encoders.bert_encoder import BERTEncoder
from texar.tf.utils.test import pretrained_test


class BERTEncoderTest(tf.test.TestCase):
    """Tests :class:`~texar.tf.modules.BERTEncoder` class.
    """

    @pretrained_test
    def test_model_loading(self):
        r"""Tests model loading functionality."""

        inputs = tf.placeholder(dtype=tf.int32, shape=[None, None])

        for pretrained_model_name in BERTEncoder.available_checkpoints():
            encoder = BERTEncoder(pretrained_model_name=pretrained_model_name)
            _, _ = encoder(inputs)

    @pretrained_test
    def test_hparams(self):
        """Tests the priority of the encoder arch parameter.
        """

        inputs = tf.placeholder(dtype=tf.int32, shape=[None, None])

        # case 1: set "pretrained_mode_name" by constructor argument
        hparams = {
            "pretrained_model_name": "bert-large-uncased",
        }
        encoder = BERTEncoder(pretrained_model_name="bert-base-uncased",
                              hparams=hparams)
        _, _ = encoder(inputs)
        self.assertEqual(encoder.hparams.encoder.num_blocks, 12)

        # case 2: set "pretrained_mode_name" by hparams
        hparams = {
            "pretrained_model_name": "bert-large-uncased",
            "encoder": {
                "num_blocks": 6
            }
        }
        encoder = BERTEncoder(hparams=hparams)
        _, _ = encoder(inputs)
        self.assertEqual(encoder.hparams.encoder.num_blocks, 24)

        # case 3: set to None in both hparams and constructor argument
        hparams = {
            "pretrained_model_name": None,
            "encoder": {
                "num_blocks": 6
            },
        }
        encoder = BERTEncoder(hparams=hparams)
        _, _ = encoder(inputs)
        self.assertEqual(encoder.hparams.encoder.num_blocks, 6)

        # case 4: using default hparams
        encoder = BERTEncoder()
        _, _ = encoder(inputs)
        self.assertEqual(encoder.hparams.encoder.num_blocks, 12)

    @pretrained_test
    def test_trainable_variables(self):
        """Tests the functionality of automatically collecting trainable
        variables.
        """

        inputs = tf.placeholder(dtype=tf.int32, shape=[None, None])

        # case 1: bert base
        encoder = BERTEncoder()
        _, _ = encoder(inputs)
        self.assertEqual(len(encoder.trainable_variables), 3 + 2 + 12 * 16 + 2)

        # case 2: bert large
        hparams = {
            "pretrained_model_name": "bert-large-uncased"
        }
        encoder = BERTEncoder(hparams=hparams)
        _, _ = encoder(inputs)
        self.assertEqual(len(encoder.trainable_variables), 3 + 2 + 24 * 16 + 2)

        # case 3: self-designed bert
        hparams = {
            "encoder": {
                "num_blocks": 6,
            },
            "pretrained_model_name": None
        }
        encoder = BERTEncoder(hparams=hparams)
        _, _ = encoder(inputs)
        self.assertEqual(len(encoder.trainable_variables), 3 + 2 + 6 * 16 + 2)

    def test_encode(self):
        """Tests encoding.
        """
        # case 1: bert base
        hparams = {
            "pretrained_model_name": None
        }
        encoder = BERTEncoder(hparams=hparams)

        max_time = 8
        batch_size = 16
        inputs = tf.random_uniform([batch_size, max_time],
                                   maxval=30521, dtype=tf.int32)
        outputs, pooled_output = encoder(inputs)

        outputs_dim = encoder.hparams.encoder.dim
        pooled_output_dim = encoder.hparams.hidden_size
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs_, pooled_output_ = sess.run([outputs, pooled_output])
            self.assertEqual(outputs_.shape, (batch_size,
                                              max_time, outputs_dim))
            self.assertEqual(pooled_output_.shape, (batch_size,
                                                    pooled_output_dim))

        # case 2: self-designed bert
        hparams = {
            "hidden_size": 100,
            "pretrained_model_name": None
        }
        encoder = BERTEncoder(hparams=hparams)

        max_time = 8
        batch_size = 16
        inputs = tf.random_uniform([batch_size, max_time],
                                   maxval=30521, dtype=tf.int32)
        outputs, pooled_output = encoder(inputs)

        outputs_dim = encoder.hparams.encoder.dim
        pooled_output_dim = encoder.hparams.hidden_size
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs_, pooled_output_ = sess.run([outputs, pooled_output])
            self.assertEqual(outputs_.shape, (batch_size,
                                              max_time, outputs_dim))
            self.assertEqual(pooled_output_.shape,
                             (batch_size, pooled_output_dim))


if __name__ == "__main__":
    tf.test.main()
