"""
Unit tests for bert utils.
"""

import os
import tensorflow as tf

from texar.tf.modules.pretrained.bert_utils import \
    load_pretrained_bert, transform_bert_to_texar_config


class BertUtilsTest(tf.test.TestCase):
    r"""Tests bert utils.
    """

    def test_load_pretrained_model_AND_transform_bert_to_texar_config(self):

        pretrained_model_dir = load_pretrained_bert(
            pretrained_model_name="bert-base-uncased")

        info = list(os.walk(pretrained_model_dir))
        _, _, files = info[0]
        self.assertIn('bert_model.ckpt.meta', files)
        self.assertIn('bert_model.ckpt.data-00000-of-00001', files)
        self.assertIn('bert_model.ckpt.index', files)
        self.assertIn('bert_config.json', files)

        model_config = transform_bert_to_texar_config(pretrained_model_dir)

        expected_config = {
            'hidden_size': 768,
            'embed': {'name': 'word_embeddings', 'dim': 768},
            'vocab_size': 30522,
            'segment_embed': {'name': 'token_type_embeddings', 'dim': 768},
            'type_vocab_size': 2,
            'position_embed': {'name': 'position_embeddings', 'dim': 768},
            'position_size': 512,
            'encoder': {
                  'name': 'encoder',
                  'embedding_dropout': 0.1,
                  'num_blocks': 12,
                  'multihead_attention': {
                      'use_bias': True,
                      'num_units': 768,
                      'num_heads': 12,
                      'output_dim': 768,
                      'dropout_rate': 0.1,
                      'name': 'self'},
                  'residual_dropout': 0.1,
                  'dim': 768,
                  'use_bert_config': True,
                  'poswise_feedforward': {
                      'layers': [{
                          'type': 'Dense',
                          'kwargs': {
                              'name': 'intermediate',
                              'units': 3072,
                              'activation': 'gelu',
                              'use_bias': True
                          }
                      },
                          {
                          'type': 'Dense',
                          'kwargs': {
                              'name': 'output',
                              'units': 768,
                              'activation': None,
                              'use_bias': True}
                      }]
                  }
            }
        }

        self.assertDictEqual(model_config, expected_config)


if __name__ == "__main__":
    tf.test.main()
