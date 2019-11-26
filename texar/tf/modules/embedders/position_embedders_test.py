"""
Unit tests for position embedders.
"""

import numpy as np

import tensorflow as tf

from texar.tf.modules.embedders.position_embedders import (
    PositionEmbedder, SinusoidsPositionEmbedder)


class PositionEmbedderTest(tf.test.TestCase):
    """Tests position embedder.
    """

    def _test_position_embedder(self, hparams):
        """Tests :class:`texar.tf.modules.PositionEmbedder`.
        """
        pos_size = 100
        embedder = PositionEmbedder(
            position_size=pos_size, hparams=hparams)
        inputs = tf.ones([64, 16], dtype=tf.int32)
        outputs = embedder(inputs)

        emb_dim = embedder.dim
        if not isinstance(emb_dim, (list, tuple)):
            emb_dim = [emb_dim]

        hparams_dim = hparams["dim"]
        if not isinstance(hparams["dim"], (list, tuple)):
            hparams_dim = [hparams["dim"]]

        self.assertEqual(outputs.shape, [64, 16] + emb_dim)
        self.assertEqual(emb_dim, hparams_dim)
        self.assertEqual(embedder.position_size, 100)
        self.assertEqual(len(embedder.trainable_variables), 1)

        seq_length = tf.random.uniform([64], maxval=pos_size, dtype=tf.int32)
        outputs = embedder(None, sequence_length=seq_length)
        self.assertEqual(outputs.shape, (64, tf.reduce_max(seq_length)) +
                         tuple(emb_dim))

    def test_embedder(self):
        """Tests various embedders.
        """
        # no dropout
        hparams = {"dim": 1024, "dropout_rate": 0}
        self._test_position_embedder(hparams)

        hparams = {"dim": [1024], "dropout_rate": 0}
        self._test_position_embedder(hparams)

        hparams = {"dim": [1024, 10], "dropout_rate": 0}
        self._test_position_embedder(hparams)

        # dropout with default strategy
        hparams = {"dim": 1024, "dropout_rate": 0.3}
        self._test_position_embedder(hparams)

        hparams = {"dim": [1024], "dropout_rate": 0.3}
        self._test_position_embedder(hparams)

        hparams = {"dim": [1024, 10], "dropout_rate": 0.3}
        self._test_position_embedder(hparams)

        # dropout with different strategies
        hparams = {"dim": 1024, "dropout_rate": 0.3,
                   "dropout_strategy": "item"}
        self._test_position_embedder(hparams)

        hparams = {"dim": [1024], "dropout_rate": 0.3,
                   "dropout_strategy": "item"}
        self._test_position_embedder(hparams)

        hparams = {"dim": [1024, 10], "dropout_rate": 0.3,
                   "dropout_strategy": "item"}
        self._test_position_embedder(hparams)

        hparams = {"dim": 1024, "dropout_rate": 0.3,
                   "dropout_strategy": "item_type"}
        self._test_position_embedder(hparams)

        hparams = {"dim": [1024], "dropout_rate": 0.3,
                   "dropout_strategy": "item_type"}
        self._test_position_embedder(hparams)

        hparams = {"dim": [1024, 10], "dropout_rate": 0.3,
                   "dropout_strategy": "item_type"}
        self._test_position_embedder(hparams)

    def test_sinusoids_position_embedder(self):
        """Tests :class:`texar.tf.modules.SinusoidsPositionEmbedder`.
        """
        position_size = 64
        input_size = (23, 18)
        hparams = {'dim': 513}  # use odd dimension to ensure padding correct
        embedder = SinusoidsPositionEmbedder(position_size, hparams=hparams)
        inputs = tf.random.uniform(shape=input_size, maxval=position_size - 1,
                                   dtype=tf.dtypes.int64)
        outputs = embedder(inputs)
        self.assertEqual(outputs.shape, input_size + (hparams['dim'],))

        embedder_no_cache = SinusoidsPositionEmbedder(
            None, hparams={**hparams, 'cache_embeddings': False})
        wide_inputs = tf.random.uniform(minval=-position_size,
                                        maxval=position_size * 2,
                                        shape=input_size,
                                        dtype=tf.dtypes.int64)
        wide_outputs = embedder_no_cache(wide_inputs)
        self.assertEqual(wide_outputs.shape, input_size + (hparams['dim'],))
        no_cache_outputs = embedder_no_cache(inputs)
        np.testing.assert_array_equal(outputs, no_cache_outputs)


if __name__ == "__main__":
    tf.test.main()
