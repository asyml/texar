"""
Unit tests for Transformer encoder.
"""

import tensorflow as tf

from texar.tf.modules.encoders.transformer_encoder import TransformerEncoder


class TransformerEncoderTest(tf.test.TestCase):

    def setUp(self):
        self._batch_size = 2
        self._emb_dim = 512
        self._max_time = 7

    def test_trainable_variables(self):
        inputs = tf.random.uniform(
            [self._batch_size, self._max_time, self._emb_dim],
            dtype=tf.float32)

        sequence_length = tf.random.uniform([self._batch_size],
                                            maxval=self._max_time,
                                            dtype=tf.int32)

        encoder = TransformerEncoder()

        _ = encoder(inputs=inputs, sequence_length=sequence_length)

        # 6 blocks
        # -self multihead_attention: 4 dense without bias + 2 layer norm vars
        # -poswise_network: Dense with bias, Dense with bias + 2 layer norm vars
        # 2 output layer norm vars

        self.assertEqual(len(encoder.trainable_variables), 74)

        hparams = {"use_bert_config": True}
        encoder = TransformerEncoder(hparams=hparams)

        # 6 blocks
        # -self multihead_attention: 4 dense without bias + 2 layer norm vars
        # -poswise_network: Dense with bias, Dense with bias + 2 layer norm vars
        # -output: 2 layer norm vars
        # 2 input layer norm vars
        _ = encoder(inputs=inputs, sequence_length=sequence_length)

        self.assertEqual(len(encoder.trainable_variables), 74)

    def test_encode(self):
        inputs = tf.random.uniform(
            [self._batch_size, self._max_time, self._emb_dim],
            dtype=tf.float32)

        sequence_length = tf.random.uniform([self._batch_size],
                                            maxval=self._max_time,
                                            dtype=tf.int32)

        encoder = TransformerEncoder()
        outputs = encoder(inputs=inputs, sequence_length=sequence_length)
        self.assertEqual(outputs.shape.as_list(), [self._batch_size,
                                                   self._max_time,
                                                   self._emb_dim])

        hparams = {"use_bert_config": True}
        encoder = TransformerEncoder(hparams=hparams)
        outputs = encoder(inputs=inputs, sequence_length=sequence_length)
        self.assertEqual(outputs.shape.as_list(), [self._batch_size,
                                                   self._max_time,
                                                   self._emb_dim])


if __name__ == "__main__":
    tf.test.main()
