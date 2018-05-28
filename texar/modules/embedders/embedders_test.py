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
from texar.modules.embedders.position_embedders import PositionEmbedder
from texar.context import global_mode

class WordEmbedderTest(tf.test.TestCase):
    """Tests word embedder.
    """
    def test_word_embedder(self):
        """Tests :class:`texar.modules.WordEmbedder`.
        """
        embedder = WordEmbedder(
            vocab_size=100,
            hparams={"dim": 1024, "dropout_rate": 0.3})
        inputs = tf.ones([64, 16], dtype=tf.int32)
        outputs = embedder(inputs)
        self.assertEqual(outputs.shape, [64, 16, 1024])
        self.assertEqual(embedder.dim, 1024)
        self.assertEqual(embedder.vocab_size, 100)
        self.assertEqual(len(embedder.trainable_variables), 1)

class PositionEmbedderTest(tf.test.TestCase):
    """Tests position embedder.
    """
    def test_position_embedder(self):
        """Tests :class:`texar.modules.PositionEmbedder`.
        """
        pos_size = 100
        embedder = PositionEmbedder(
            position_size=pos_size, hparams={"dim": 1024})
        inputs = tf.random_uniform([64, 16], maxval=pos_size, dtype=tf.int32)
        outputs = embedder(positions=inputs)
        self.assertEqual(outputs.shape, [64, 16, 1024])
        self.assertEqual(embedder.dim, 1024)
        self.assertEqual(embedder.position_size, 100)
        self.assertEqual(len(embedder.trainable_variables), 1)

        seq_length = tf.random_uniform([64], maxval=pos_size, dtype=tf.int32)
        outputs = embedder(sequence_length=seq_length)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs_, max_seq_length = sess.run(
                [outputs, tf.reduce_max(seq_length)],
                feed_dict={global_mode(): tf.estimator.ModeKeys.TRAIN})
            self.assertEqual(outputs_.shape, (64, max_seq_length, 1024))

if __name__ == "__main__":
    tf.test.main()
