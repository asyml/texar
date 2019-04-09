"""
Unit tests for beam search decoding.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import tensorflow as tf
from tensorflow.contrib.seq2seq import dynamic_decode
from tensorflow.contrib.seq2seq import BeamSearchDecoder, tile_batch

import texar as tx
from texar.modules.decoders.beam_search_decode import beam_search_decode
from texar import context

# pylint: disable=no-member, too-many-instance-attributes, invalid-name
# pylint: disable=too-many-locals, too-many-arguments

class BeamSearchDecodeTest(tf.test.TestCase):
    """Tests
    :func:`texar.modules.decoders.beam_search_decode.beam_search_decode`.
    """

    def setUp(self):
        tf.test.TestCase.setUp(self)
        self._vocab_size = 10
        self._max_time = 16
        self._batch_size = 8
        self._emb_dim = 20
        self._cell_dim = 256
        self._attention_dim = self._cell_dim
        self._beam_width = 11
        self._inputs = tf.random_uniform(
            [self._batch_size, self._max_time, self._emb_dim],
            maxval=1., dtype=tf.float32)
        self._embedding = tf.random_uniform(
            [self._vocab_size, self._emb_dim], maxval=1., dtype=tf.float32)
        self._encoder_output = tf.random_uniform(
            [self._batch_size, self._max_time, 64])

    def _test_beam_search(
            self, decoder, initial_state=None, tiled_initial_state=None,
            tf_initial_state=None, beam_width_1=1, initiated=False):
        # Compare with tf built-in BeamSearchDecoder
        outputs, final_state, _ = beam_search_decode(
            decoder_or_cell=decoder,
            embedding=self._embedding,
            start_tokens=[1]*self._batch_size,
            end_token=2,
            beam_width=beam_width_1,
            max_decoding_length=20)

        self.assertIsInstance(
            outputs, tf.contrib.seq2seq.FinalBeamSearchDecoderOutput)
        self.assertIsInstance(
            final_state, tf.contrib.seq2seq.BeamSearchDecoderState)

        num_trainable_variables = len(tf.trainable_variables())
        _ = decoder(
            decoding_strategy='infer_greedy',
            embedding=self._embedding,
            start_tokens=[1]*self._batch_size,
            end_token=2,
            max_decoding_length=20)
        self.assertEqual(num_trainable_variables, len(tf.trainable_variables()))

        if tf_initial_state is None:
            tf_initial_state = decoder.cell.zero_state(
                self._batch_size * beam_width_1, tf.float32)
        beam_decoder = BeamSearchDecoder(
            cell=decoder.cell,
            embedding=self._embedding,
            start_tokens=[1]*self._batch_size,
            end_token=2,
            initial_state=tf_initial_state,
            beam_width=beam_width_1,
            output_layer=decoder.output_layer)

        outputs_1, final_state_1, _ = dynamic_decode(
            decoder=beam_decoder, maximum_iterations=20)

        ## Tests time major
        outputs_2, _, _ = beam_search_decode(
            decoder_or_cell=decoder,
            embedding=self._embedding,
            start_tokens=[1]*self._batch_size,
            end_token=2,
            beam_width=self._beam_width,
            initial_state=initial_state,
            tiled_initial_state=tiled_initial_state,
            max_decoding_length=21)
        outputs_3, _, _ = beam_search_decode(
            decoder_or_cell=decoder,
            embedding=self._embedding,
            start_tokens=[1]*self._batch_size,
            end_token=2,
            beam_width=self._beam_width,
            initial_state=initial_state,
            tiled_initial_state=tiled_initial_state,
            max_decoding_length=21,
            output_time_major=True)


        with self.test_session() as sess:
            if not initiated:
                sess.run(tf.global_variables_initializer())

            outputs_, final_state_, outputs_1_, final_state_1_ = sess.run(
                [outputs, final_state, outputs_1, final_state_1],
                feed_dict={context.global_mode():
                           tf.estimator.ModeKeys.PREDICT})

            np.testing.assert_array_equal(
                outputs_.predicted_ids, outputs_1_.predicted_ids)
            np.testing.assert_array_equal(
                outputs_.beam_search_decoder_output.scores,
                outputs_1_.beam_search_decoder_output.scores)
            np.testing.assert_array_equal(
                outputs_.beam_search_decoder_output.predicted_ids,
                outputs_1_.beam_search_decoder_output.predicted_ids)
            np.testing.assert_array_equal(
                outputs_.beam_search_decoder_output.parent_ids,
                outputs_1_.beam_search_decoder_output.parent_ids)
            np.testing.assert_array_equal(
                final_state_.log_probs, final_state_1_.log_probs)
            np.testing.assert_array_equal(
                final_state_.lengths, final_state_1_.lengths)

            outputs_2_, outputs_3_ = sess.run(
                [outputs_2, outputs_3],
                feed_dict={context.global_mode():
                           tf.estimator.ModeKeys.PREDICT})
            self.assertEqual(outputs_2_.predicted_ids.shape,
                             tuple([self._batch_size, 21, 11]))
            self.assertEqual(outputs_3_.predicted_ids.shape,
                             tuple([21, self._batch_size, 11]))

    def test_basic_rnn_decoder_beam_search(self):
        """Tests beam search with BasicRNNDecoder.
        """
        hparams = {
            "rnn_cell": {
                "kwargs": {"num_units": self._cell_dim}
            }
        }
        decoder = tx.modules.BasicRNNDecoder(
            vocab_size=self._vocab_size,
            hparams=hparams)

        self._test_beam_search(decoder)

        self._test_beam_search(
            decoder, beam_width_1=self._beam_width, initiated=True)

    def test_basic_rnn_decoder_given_initial_state(self):
        """Tests beam search with BasicRNNDecoder given initial state.
        """
        hparams = {
            "rnn_cell": {
                "kwargs": {"num_units": self._cell_dim}
            }
        }
        decoder = tx.modules.BasicRNNDecoder(
            vocab_size=self._vocab_size,
            hparams=hparams)

        # (zhiting): The beam search decoder does not generate max-length
        # samples if only one cell_state is created. Perhaps due to
        # random seed or bugs?
        cell_state = decoder.cell.zero_state(self._batch_size, tf.float32)

        self._test_beam_search(decoder, initial_state=cell_state)

        tiled_cell_state = tile_batch(cell_state, multiplier=self._beam_width)
        self._test_beam_search(
            decoder, tiled_initial_state=tiled_cell_state, initiated=True)

    def test_attention_decoder_beam_search(self):
        """Tests beam search with RNNAttentionDecoder.
        """
        seq_length = np.random.randint(
            self._max_time, size=[self._batch_size]) + 1
        encoder_values_length = tf.constant(seq_length)
        hparams = {
            "attention": {
                "kwargs": {"num_units": self._attention_dim}
            },
            "rnn_cell": {
                "kwargs": {"num_units": self._cell_dim}
            }
        }
        decoder = tx.modules.AttentionRNNDecoder(
            vocab_size=self._vocab_size,
            memory=self._encoder_output,
            memory_sequence_length=encoder_values_length,
            hparams=hparams)

        self._test_beam_search(decoder)

    def test_attention_decoder_given_initial_state(self):
        """Tests beam search with RNNAttentionDecoder given initial state.
        """
        seq_length = np.random.randint(
            self._max_time, size=[self._batch_size]) + 1
        encoder_values_length = tf.constant(seq_length)
        hparams = {
            "attention": {
                "kwargs": {"num_units": self._attention_dim}
            },
            "rnn_cell": {
                "kwargs": {"num_units": self._cell_dim}
            }
        }
        decoder = tx.modules.AttentionRNNDecoder(
            vocab_size=self._vocab_size,
            memory=self._encoder_output,
            memory_sequence_length=encoder_values_length,
            hparams=hparams)

        state = decoder.cell.zero_state(self._batch_size, tf.float32)

        cell_state = state.cell_state
        self._test_beam_search(decoder, initial_state=cell_state)

        tiled_cell_state = tile_batch(cell_state, multiplier=self._beam_width)
        self._test_beam_search(
            decoder, tiled_initial_state=tiled_cell_state, initiated=True)


if __name__ == "__main__":
    tf.test.main()
