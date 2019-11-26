"""
Unit tests for xlnet utils.
"""

import os
import tensorflow as tf

from texar.tf.modules.pretrained.xlnet import *
from texar.tf.utils.test import pretrained_test


class XLNetUtilsTest(tf.test.TestCase):
    r"""Tests XLNet utils.
    """

    @pretrained_test
    def test_load_pretrained_model_AND_transform_xlnet_to_texar_config(self):

        pretrained_model_dir = PretrainedXLNetMixin.download_checkpoint(
            pretrained_model_name="xlnet-base-cased")

        info = list(os.walk(pretrained_model_dir))
        _, _, files = info[0]
        self.assertIn('spiece.model', files)
        self.assertIn('xlnet_model.ckpt.meta', files)
        self.assertIn('xlnet_model.ckpt.data-00000-of-00001', files)
        self.assertIn('xlnet_model.ckpt.index', files)
        self.assertIn('xlnet_config.json', files)

        model_config = PretrainedXLNetMixin._transform_config(
            pretrained_model_name="xlnet-base-cased",
            cache_dir=pretrained_model_dir)

        expected_config = {
            'head_dim': 64,
            'ffn_inner_dim': 3072,
            'hidden_dim': 768,
            'activation': 'gelu',
            'num_heads': 12,
            'num_layers': 12,
            'vocab_size': 32000,
            'untie_r': True
        }

        self.assertDictEqual(model_config, expected_config)


if __name__ == "__main__":
    tf.test.main()
