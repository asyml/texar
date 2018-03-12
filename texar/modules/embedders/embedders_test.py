#
"""
Unit tests for embedders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# pylint: disable=no-member

import tensorflow as tf

from texar.modules.embedders.embedders import WordEmbedder

class WordEmbedderTest(tf.test.TestCase):
    """Tests word embedder.
    """
    def test_word_embedder(self):
        """Tests :class:`texar.modules.WordEmbedder`.
        """
        embedder = WordEmbedder(vocab_size=100, hparams={"dim": 1024})
        inputs = tf.ones([64, 16], dtype=tf.int32)
        outputs = embedder(inputs)
        self.assertEqual(outputs.shape, [64, 16, 1024])
        self.assertEqual(embedder.dim, 1024)
        self.assertEqual(embedder.vocab_size, 100)
        self.assertEqual(len(embedder.trainable_variables), 1)

if __name__ == "__main__":
    tf.test.main()
