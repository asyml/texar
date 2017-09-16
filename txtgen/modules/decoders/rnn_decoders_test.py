#
"""
Unit tests for RNN decoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import tensorflow as tf
from tensorflow.contrib.seq2seq import BasicDecoderOutput

from txtgen.modules.decoders.rnn_decoders import BasicRNNDecoder
from txtgen.modules.decoders.rnn_decoder_helpers import make_helper
from txtgen import context

class BasicRNNDecoderTest(tf.test.TestCase):
    """Test basic RNN decoder.
    """

    def setUp(self):
        tf.test.TestCase.setUp(self)
        self._vocab_size = 4
        self._max_time = 8
        self._batch_size = 16
        self._inputs = tf.random_uniform([self._batch_size, self._max_time],
                                         maxval=self._vocab_size,
                                         dtype=tf.int32)

    def test_decode_train(self):
        """Tests decoding in training mode.
        """
        decoder = BasicRNNDecoder(vocab_size=self._vocab_size)

        helper_train = make_helper(
            decoder.hparams.helper_train.type,
            inputs=self._inputs,
            sequence_length=[self._max_time]*self._batch_size,
            embedding=decoder.embedding,
            **decoder.hparams.helper_train.kwargs.todict())

        outputs, final_state, sequence_lengths = decoder(
            helper_train, decoder.cell.zero_state(self._batch_size, tf.float32))

        # 5 trainable variables: embedding, cell-kernel, cell-bias,
        # fc-layer-weights, fc-layer-bias
        self.assertEqual(len(decoder.trainable_variables), 5)

        cell_dim = decoder.hparams.rnn_cell.cell.kwargs.num_units
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs_, final_state_, sequence_lengths_ = sess.run(
                [outputs, final_state, sequence_lengths],
                feed_dict={context.is_train(): True})
            self.assertIsInstance(outputs_, BasicDecoderOutput)
            self.assertEqual(
                outputs_.rnn_output.shape,
                (self._batch_size, self._max_time, self._vocab_size))
            self.assertEqual(
                outputs_.sample_id.shape, (self._batch_size, self._max_time))
            self.assertEqual(final_state_[0].shape,
                             (self._batch_size, cell_dim))
            np.testing.assert_array_equal(
                sequence_lengths_,
                [self._max_time]*self._batch_size)


    def test_decode_infer(self):
        """Tests decoding in inferencee mode.
        """
        decoder = BasicRNNDecoder(vocab_size=self._vocab_size)

        helper_infer = make_helper(
            decoder.hparams.helper_infer.type,
            embedding=decoder.embedding,
            start_tokens=[self._vocab_size-2]*self._batch_size,
            end_token=self._vocab_size-1,
            **decoder.hparams.helper_train.kwargs.todict())

        outputs, final_state, sequence_lengths = decoder(
            helper_infer, decoder.cell.zero_state(self._batch_size, tf.float32))

        # 5 trainable variables: embedding, cell-kernel, cell-bias,
        # fc-layer-weights, fc-layer-bias
        self.assertEqual(len(decoder.trainable_variables), 5)

        cell_dim = decoder.hparams.rnn_cell.cell.kwargs.num_units
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs_, final_state_, sequence_lengths_ = sess.run(
                [outputs, final_state, sequence_lengths],
                feed_dict={context.is_train(): True})
            self.assertIsInstance(outputs_, BasicDecoderOutput)
            max_length = max(sequence_lengths_)
            self.assertEqual(
                outputs_.rnn_output.shape,
                (self._batch_size, max_length, self._vocab_size))
            self.assertEqual(
                outputs_.sample_id.shape, (self._batch_size, max_length))
            self.assertEqual(final_state_[0].shape,
                             (self._batch_size, cell_dim))

if __name__ == "__main__":
    tf.test.main()

