#
"""
Unit tests for embedder utils.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# pylint: disable=no-member

import tensorflow as tf

from texar.modules.embedders import embedder_utils

class GetEmbeddingTest(tf.test.TestCase):
    """Tests embedding creator.
    """
    def test_get_embedding(self):
        """Tests :func:`~texar.modules.embedder.embedder_utils.get_embedding`.
        """
        vocab_size = 100
        emb = embedder_utils.get_embedding(num_embeds=vocab_size)
        self.assertEqual(emb.shape[0].value, vocab_size)
        self.assertEqual(emb.shape[1].value,
                         embedder_utils.default_embedding_hparams()["dim"])

        hparams = {
            "initializer": {
                "type": tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
            },
            "regularizer": {
                "type": tf.keras.regularizers.L1L2(0.1, 0.1)
            }
        }
        emb = embedder_utils.get_embedding(
            hparams=hparams, num_embeds=vocab_size,
            variable_scope='embedding_2')
        self.assertEqual(emb.shape[0].value, vocab_size)
        self.assertEqual(emb.shape[1].value,
                         embedder_utils.default_embedding_hparams()["dim"])


if __name__ == "__main__":
    tf.test.main()
