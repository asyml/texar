#
"""
Unit tests for Transformer decodre.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf

from texar.modules.decoders.transformer_decoders import TransformerDecoder
from texar.modules.decoders.transformer_decoders import TransformerDecoderOutput
from texar.modules.decoders import tf_helpers as tx_helper

# pylint: disable=too-many-instance-attributes


class TransformerDecoderTest(tf.test.TestCase):
    """Tests :class:`~texar.modules.TransformerDecoder`
    """

    def setUp(self):
        tf.test.TestCase.setUp(self)
        self._vocab_size = 15
        self._batch_size = 2
        self._max_time = 10
        self._emb_dim = 512
        self._max_decode_len = 16
        self._inputs = tf.random_uniform(
            [self._batch_size, self._max_time, self._emb_dim],
            maxval=1, dtype=tf.float32)

        self._memory = tf.random_uniform(
            [self._batch_size, self._max_time, self._emb_dim],
            maxval=1, dtype=tf.float32)
        self._memory_sequence_length = tf.random_uniform(
            [self._batch_size], maxval=self._max_time, dtype=tf.int32)

        self._embedding = tf.random_uniform(
            [self._vocab_size, self._emb_dim], maxval=1, dtype=tf.float32)
        self._pos_embedding = tf.random_uniform(
            [self._max_decode_len, self._emb_dim], maxval=1, dtype=tf.float32)

        def _embedding_fn(x, y):
            x_emb = tf.nn.embedding_lookup(self._embedding, x)
            y_emb = tf.nn.embedding_lookup(self._pos_embedding, y)
            return x_emb * self._emb_dim ** 0.5 + y_emb
        self._embedding_fn = _embedding_fn

        self._output_layer = tf.random_uniform(
            [self._emb_dim, self._vocab_size], maxval=1, dtype=tf.float32)

        self._start_tokens = tf.fill([self._batch_size], 1)
        self._end_token = 2
        self.max_decoding_length = self._max_time

        _context = [[3, 4, 5, 2, 0], [4, 3, 5, 7, 2]]
        _context_length = [4, 5]
        self._context = tf.Variable(_context)
        self._context_length = tf.Variable(_context_length)

    def test_output_layer(self):
        decoder = TransformerDecoder(vocab_size=self._vocab_size,
                                     output_layer=None)
        self.assertIsInstance(decoder, TransformerDecoder)

        decoder = TransformerDecoder(output_layer=tf.identity)
        self.assertIsInstance(decoder, TransformerDecoder)

        tensor = tf.random_uniform(
            [self._emb_dim, self._vocab_size], maxval=1, dtype=tf.float32
        )
        decoder = TransformerDecoder(output_layer=tensor)
        self.assertIsInstance(decoder, TransformerDecoder)
        self.assertEqual(decoder.vocab_size, self._vocab_size)

    def test_decode_train(self):
        """Tests train_greedy
        """
        decoder = TransformerDecoder(
            vocab_size=self._vocab_size,
            output_layer=self._output_layer
        )
        # 6 blocks
        # -self multihead_attention: 4 dense without bias + 2 layer norm vars
        # -encdec multihead_attention: 4 dense without bias + 2 layer norm vars
        # -poswise_network: Dense with bias, Dense with bias + 2 layer norm vars
        # 2 layer norm vars
        outputs = decoder(memory=self._memory,
                          memory_sequence_length=self._memory_sequence_length,
                          memory_attention_bias=None,
                          inputs=self._inputs,
                          decoding_strategy='train_greedy',
                          mode=tf.estimator.ModeKeys.TRAIN)
        self.assertEqual(len(decoder.trainable_variables), 110)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs_ = sess.run(outputs)

            self.assertIsInstance(outputs_, TransformerDecoderOutput)

    def test_decode_infer_greedy(self):
        """Tests train_greedy
        """
        decoder = TransformerDecoder(
            vocab_size=self._vocab_size,
            output_layer=self._output_layer
        )
        helper = tx_helper.GreedyEmbeddingHelper(
            self._embedding_fn, self._start_tokens, self._end_token)

        outputs, length = decoder(
            memory=self._memory,
            memory_sequence_length=self._memory_sequence_length,
            memory_attention_bias=None,
            inputs=None,
            helper=helper,
            max_decoding_length=self._max_decode_len,
            mode=tf.estimator.ModeKeys.PREDICT)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs_ = sess.run(outputs)
            self.assertIsInstance(outputs_, TransformerDecoderOutput)

    def test_infer_greedy_with_context_without_memory(self):
        """Tests train_greedy with context
        """
        decoder = TransformerDecoder(
            vocab_size=self._vocab_size,
            output_layer=self._output_layer
        )
        helper = tx_helper.GreedyEmbeddingHelper(
            self._embedding_fn, self._start_tokens, self._end_token)

        outputs, length = decoder(
            memory=None,
            memory_sequence_length=None,
            memory_attention_bias=None,
            inputs=None,
            decoding_strategy='infer_greedy',
            helper=helper,
            context=self._context,
            context_sequence_length=self._context_length,
            end_token=self._end_token,
            max_decoding_length=self._max_decode_len,
            mode=tf.estimator.ModeKeys.PREDICT)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs_ = sess.run(outputs)
            self.assertIsInstance(outputs_, TransformerDecoderOutput)

    def test_decode_infer_sample(self):
        """Tests infer_sample
        """
        decoder = TransformerDecoder(
            vocab_size=self._vocab_size,
            output_layer=self._output_layer
        )
        helper = tx_helper.SampleEmbeddingHelper(
            self._embedding_fn, self._start_tokens, self._end_token)

        outputs, length = decoder(
            memory=self._memory,
            memory_sequence_length=self._memory_sequence_length,
            memory_attention_bias=None,
            inputs=None,
            helper=helper,
            max_decoding_length=self._max_decode_len,
            mode=tf.estimator.ModeKeys.PREDICT)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs_ = sess.run(outputs)
            self.assertIsInstance(outputs_, TransformerDecoderOutput)

    def test_beam_search(self):
        """Tests beam_search
        """
        decoder = TransformerDecoder(
            vocab_size=self._vocab_size,
            output_layer=self._output_layer
        )

        outputs = decoder(
            memory=self._memory,
            memory_sequence_length=self._memory_sequence_length,
            memory_attention_bias=None,
            inputs=None,
            embedding=self._embedding_fn,
            beam_width=5,
            start_tokens=self._start_tokens,
            end_token=self._end_token,
            max_decoding_length=self._max_decode_len,
            mode=tf.estimator.ModeKeys.PREDICT
        )

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs_ = sess.run(outputs)
            self.assertEqual(outputs_['log_prob'].shape,
                             (self._batch_size, 5))
            self.assertEqual(outputs_['sample_id'].shape,
                             (self._batch_size, self._max_decode_len, 5))

    def test_greedy_embedding_helper(self):
        """Tests with tf.contrib.seq2seq.GreedyEmbeddingHelper
        """
        decoder = TransformerDecoder(
            vocab_size=self._vocab_size,
            output_layer=self._output_layer
        )
        helper = tx_helper.GreedyEmbeddingHelper(
            self._embedding, self._start_tokens, self._end_token)
        outputs, length = decoder(
            memory=self._memory,
            memory_sequence_length=self._memory_sequence_length,
            memory_attention_bias=None,
            helper=helper,
            max_decoding_length=self._max_decode_len,
            mode=tf.estimator.ModeKeys.PREDICT)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs_ = sess.run(outputs)
            self.assertIsInstance(outputs_, TransformerDecoderOutput)


if __name__ == "__main__":
    tf.test.main()
