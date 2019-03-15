"""
Unit tests for decoder helpers.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf

from texar.modules.decoders.rnn_decoder_helpers import \
        SoftmaxEmbeddingHelper, GumbelSoftmaxEmbeddingHelper
from texar.modules.decoders.tf_helpers import GreedyEmbeddingHelper
from texar.modules.decoders.rnn_decoders import BasicRNNDecoder
from texar.modules.embedders.embedders import WordEmbedder
from texar.modules.embedders.position_embedders import PositionEmbedder

# pylint: disable=no-member, too-many-locals, too-many-instance-attributes
# pylint: disable=too-many-arguments, protected-access, redefined-variable-type

class HelpersTest(tf.test.TestCase):
    """Tests :class:`~texar.modules.decoders.rnn_decoders.BasicRNNDecoder`.
    """

    def setUp(self):
        tf.test.TestCase.setUp(self)
        self._batch_size = 16
        self._vocab_size = 4
        self._start_tokens = [self._vocab_size-2]*self._batch_size
        self._end_token = self._vocab_size-1
        self._max_time = 8
        self._emb_dim = 100
        self._inputs = tf.random_uniform(
            [self._batch_size, self._max_time, self._emb_dim],
            maxval=1., dtype=tf.float32)
        self._embedding = tf.random_uniform(
            [self._vocab_size, self._emb_dim], maxval=1., dtype=tf.float32)
        self._max_seq_length = 10

    def test_softmax_embedding_helpers(self):
        """Tests softmax helpers.
        """

        def _test_fn(helper):
            _, next_inputs, _ = helper.next_inputs(
                time=1,
                outputs=tf.ones([self._batch_size, self._vocab_size]),# Not used
                state=None, # Not used
                sample_ids=tf.ones([self._batch_size, self._vocab_size]))

            self.assertEqual(helper.sample_ids_shape,
                             tf.TensorShape(self._vocab_size))
            self.assertEqual(next_inputs.get_shape(),
                             tf.TensorShape([self._batch_size, self._emb_dim]))

            # Test in an RNN decoder
            output_layer = tf.layers.Dense(self._vocab_size)
            decoder = BasicRNNDecoder(vocab_size=self._vocab_size,
                                      output_layer=output_layer)
            outputs, final_state, sequence_lengths = decoder(
                helper=helper, max_decoding_length=self._max_seq_length)

            cell_dim = decoder.hparams.rnn_cell.kwargs.num_units
            with self.test_session() as sess:
                sess.run(tf.global_variables_initializer())
                outputs_, final_state_, sequence_lengths_ = sess.run(
                    [outputs, final_state, sequence_lengths])
                max_length = max(sequence_lengths_)
                self.assertEqual(
                    outputs_.logits.shape,
                    (self._batch_size, max_length, self._vocab_size))
                self.assertEqual(
                    outputs_.sample_id.shape,
                    (self._batch_size, max_length, self._vocab_size))
                self.assertEqual(final_state_[0].shape,
                                 (self._batch_size, cell_dim))

        # SoftmaxEmbeddingHelper

        # case-(1)
        helper = SoftmaxEmbeddingHelper(
            self._embedding, self._start_tokens, self._end_token, 0.7)
        _test_fn(helper)

        # case-(2)
        embedder = WordEmbedder(self._embedding)
        helper = SoftmaxEmbeddingHelper(
            embedder, self._start_tokens, self._end_token, 0.7,
            embedding_size=self._vocab_size)
        _test_fn(helper)

        # case-(3)
        word_embedder = WordEmbedder(self._embedding)
        pos_embedder = PositionEmbedder(position_size=self._max_seq_length)

        def _emb_fn(soft_ids, times):
            return word_embedder(soft_ids=soft_ids) + pos_embedder(times)
        helper = SoftmaxEmbeddingHelper(
            _emb_fn, self._start_tokens, self._end_token, 0.7,
            embedding_size=self._vocab_size)
        _test_fn(helper)

        # GumbelSoftmaxEmbeddingHelper

        # case-(1)
        helper = GumbelSoftmaxEmbeddingHelper(
            self._embedding, self._start_tokens, self._end_token, 0.7)
        _test_fn(helper)


    def test_infer_helpers(self):
        """Tests inference helpers.
        """

        def _test_fn(helper):
            _, next_inputs, _ = helper.next_inputs(
                time=1,
                outputs=tf.ones([self._batch_size, self._vocab_size]),# Not used
                state=None, # Not used
                sample_ids=tf.ones([self._batch_size], dtype=tf.int32))

            self.assertEqual(helper.sample_ids_shape,
                             tf.TensorShape([]))
            self.assertEqual(next_inputs.get_shape(),
                             tf.TensorShape([self._batch_size, self._emb_dim]))

            # Test in an RNN decoder
            output_layer = tf.layers.Dense(self._vocab_size)
            decoder = BasicRNNDecoder(vocab_size=self._vocab_size,
                                      output_layer=output_layer)
            outputs, final_state, sequence_lengths = decoder(
                helper=helper, max_decoding_length=self._max_seq_length)

            cell_dim = decoder.hparams.rnn_cell.kwargs.num_units
            with self.test_session() as sess:
                sess.run(tf.global_variables_initializer())
                outputs_, final_state_, sequence_lengths_ = sess.run(
                    [outputs, final_state, sequence_lengths])
                max_length = max(sequence_lengths_)
                self.assertEqual(
                    outputs_.logits.shape,
                    (self._batch_size, max_length, self._vocab_size))
                self.assertEqual(
                    outputs_.sample_id.shape, (self._batch_size, max_length))
                self.assertEqual(final_state_[0].shape,
                                 (self._batch_size, cell_dim))

        # case-(1)
        helper = GreedyEmbeddingHelper(
            self._embedding, self._start_tokens, self._end_token)
        _test_fn(helper)

        # case-(2)
        embedder = WordEmbedder(self._embedding)
        helper = GreedyEmbeddingHelper(
            embedder, self._start_tokens, self._end_token)
        _test_fn(helper)

        # case-(3)
        word_embedder = WordEmbedder(self._embedding)
        pos_embedder = PositionEmbedder(position_size=self._max_seq_length)

        def _emb_fn(ids, times):
            return word_embedder(ids) + pos_embedder(times)
        helper = GreedyEmbeddingHelper(
            _emb_fn, self._start_tokens, self._end_token)
        _test_fn(helper)


if __name__ == "__main__":
    tf.test.main()
