#
"""
Unit tests for XLNet encoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf

from texar.modules.encoders.xlnet_encoders import XLNetEncoder


class XLNetEncoderTest(tf.test.TestCase):
    """Tests :class:`~texar.modules.BertEncoder` class.
    """

    def test_hparams(self):
        """Tests the priority of the encoder architecture parameter.
        """

        inputs = tf.placeholder(dtype=tf.int32, shape=[None, None])

        # case 1: set "pretrained_mode_name" by constructor argument
        hparams = {
            "pretrained_model_name": "xlnet-large-cased",
        }
        encoder = XLNetEncoder(pretrained_model_name="xlnet-large-cased",
                               hparams=hparams)
        encoder(inputs, is_training=False)
        self.assertEqual(len(encoder.attn_layers), 24)
        self.assertEqual(len(encoder.ff_layers), 24)

        # case 2: set "pretrained_mode_name" by hparams
        """hparams = {
            "pretrained_model_name": "bert-large-uncased",
            "encoder": {
                "num_blocks": 6
            }
        }
        encoder = XLNetEncoder(hparams=hparams)
        _, _ = encoder(inputs)
        self.assertEqual(encoder.hparams.encoder.num_blocks, 24)

        # case 3: set to None in both hparams and constructor argument
        hparams = {
            "pretrained_model_name": None,
            "encoder": {
                "num_blocks": 6
            },
        }
        encoder = XLNetEncoder(hparams=hparams)
        _, _ = encoder(inputs)
        self.assertEqual(encoder.hparams.encoder.num_blocks, 6)

        # case 4: using default hparams
        encoder = XLNetEncoder()
        _, _ = encoder(inputs)
        self.assertEqual(encoder.hparams.encoder.num_blocks, 12)"""


if __name__ == "__main__":
    tf.test.main()
