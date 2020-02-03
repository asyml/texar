"""
Unit tests for BERT encoders.
"""

import tensorflow as tf

from texar.tf.modules.encoders.bert_encoder import BERTEncoder
from texar.tf.utils.test import pretrained_test


class BERTEncoderTest(tf.test.TestCase):
    """Tests :class:`~texar.tf.modules.BERTEncoder` class.
    """

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
        inputs = tf.random.uniform([batch_size, max_time],
                                   maxval=30521, dtype=tf.int32)
        outputs, pooled_output = encoder(inputs)

        outputs_dim = encoder.hparams.encoder.dim
        pooled_output_dim = encoder.hparams.hidden_size

        self.assertEqual(outputs.shape, (batch_size, max_time, outputs_dim))
        self.assertEqual(pooled_output.shape, (batch_size, pooled_output_dim))

        # case 2: self-designed bert
        hparams = {
            "hidden_size": 100,
            "pretrained_model_name": None
        }
        encoder = BERTEncoder(hparams=hparams)

        max_time = 8
        batch_size = 16
        inputs = tf.random.uniform([batch_size, max_time],
                                   maxval=30521, dtype=tf.int32)
        outputs, pooled_output = encoder(inputs)

        outputs_dim = encoder.hparams.encoder.dim
        pooled_output_dim = encoder.hparams.hidden_size

        self.assertEqual(outputs.shape, (batch_size, max_time, outputs_dim))
        self.assertEqual(pooled_output.shape, (batch_size, pooled_output_dim))


if __name__ == "__main__":
    tf.test.main()
