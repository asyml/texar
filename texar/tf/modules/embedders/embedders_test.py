"""
Unit tests for embedders.
"""

import numpy as np

import tensorflow as tf

from texar.tf.modules.embedders.embedders import WordEmbedder


class EmbedderTest(tf.test.TestCase):
    """Tests parameterized embedder.
    """

    def _test_word_embedder(self, hparams):
        """Tests :class:`texar.tf.modules.WordEmbedder`.
        """
        embedder = WordEmbedder(
            vocab_size=100, hparams=hparams)

        inputs = tf.ones([64, 16], dtype=tf.int32)
        outputs = embedder(inputs)

        inputs_soft = tf.ones([64, 16, embedder.vocab_size], dtype=tf.float32)
        outputs_soft = embedder(soft_ids=inputs_soft)

        emb_dim = embedder.dim
        if not isinstance(emb_dim, (list, tuple)):
            emb_dim = [emb_dim]

        hparams_dim = hparams["dim"]
        if not isinstance(hparams["dim"], (list, tuple)):
            hparams_dim = [hparams["dim"]]

        self.assertEqual(outputs.shape, [64, 16] + emb_dim)
        self.assertEqual(outputs_soft.shape, [64, 16] + emb_dim)
        self.assertEqual(emb_dim, hparams_dim)
        self.assertEqual(embedder.vocab_size, 100)
        self.assertEqual(len(embedder.trainable_variables), 1)

        self.assertEqual(outputs.shape, (64, 16) + tuple(emb_dim))
        self.assertEqual(outputs_soft.shape, (64, 16) + tuple(emb_dim))

    def test_embedder(self):
        """Tests various embedders.
        """
        # no dropout
        hparams = {"dim": 1024, "dropout_rate": 0}
        self._test_word_embedder(hparams)

        hparams = {"dim": [1024], "dropout_rate": 0}
        self._test_word_embedder(hparams)

        hparams = {"dim": [1024, 10], "dropout_rate": 0}
        self._test_word_embedder(hparams)

        # dropout with default strategy
        hparams = {"dim": 1024, "dropout_rate": 0.3}
        self._test_word_embedder(hparams)

        hparams = {"dim": [1024], "dropout_rate": 0.3}
        self._test_word_embedder(hparams)

        hparams = {"dim": [1024, 10], "dropout_rate": 0.3}
        self._test_word_embedder(hparams)

        # dropout with different strategies
        hparams = {"dim": 1024, "dropout_rate": 0.3,
                   "dropout_strategy": "item"}
        self._test_word_embedder(hparams)

        hparams = {"dim": [1024], "dropout_rate": 0.3,
                   "dropout_strategy": "item"}
        self._test_word_embedder(hparams)

        hparams = {"dim": [1024, 10], "dropout_rate": 0.3,
                   "dropout_strategy": "item"}
        self._test_word_embedder(hparams)

        hparams = {"dim": 1024, "dropout_rate": 0.3,
                   "dropout_strategy": "item_type"}
        self._test_word_embedder(hparams)

        hparams = {"dim": [1024], "dropout_rate": 0.3,
                   "dropout_strategy": "item_type"}
        self._test_word_embedder(hparams)

        hparams = {"dim": [1024, 10], "dropout_rate": 0.3,
                   "dropout_strategy": "item_type"}
        self._test_word_embedder(hparams)

    def test_embedder_multi_calls(self):
        """Tests embedders called by multiple times.
        """
        hparams = {"dim": 1024, "dropout_rate": 0.3,
                   "dropout_strategy": "item"}
        embedder = WordEmbedder(
            vocab_size=100, hparams=hparams)
        inputs = tf.ones([64, 16], dtype=tf.int32)
        outputs = embedder(inputs)

        emb_dim = embedder.dim
        if not isinstance(emb_dim, (list, tuple)):
            emb_dim = [emb_dim]
        self.assertEqual(outputs.shape, [64, 16] + emb_dim)

        # Call with inputs in a different shape
        inputs = tf.ones([64, 10, 20], dtype=tf.int32)
        outputs = embedder(inputs)

        emb_dim = embedder.dim
        if not isinstance(emb_dim, (list, tuple)):
            emb_dim = [emb_dim]
        self.assertEqual(outputs.shape, [64, 10, 20] + emb_dim)

    def test_word_embedder_soft_ids(self):
        """Tests the correctness of using soft ids.
        """
        init_value = np.expand_dims(np.arange(5), 1)
        embedder = WordEmbedder(init_value=init_value)

        ids = np.array([3])
        soft_ids = np.array([[0, 0, 0, 1, 0]])

        outputs = embedder(ids)
        soft_outputs = embedder(soft_ids=soft_ids)

        self.assertEqual(outputs, soft_outputs)


if __name__ == "__main__":
    tf.test.main()
