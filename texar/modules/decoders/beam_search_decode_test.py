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

import texar as tx
from texar.modules.decoders.beam_search_decode import beam_search_decode
from texar import context

# pylint: disable=no-member, too-many-instance-attributes, invalid-name
# pylint: disable=too-many-locals

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
        self._attention_dim = 256
        self._inputs = tf.random_uniform(
            [self._batch_size, self._max_time, self._emb_dim],
            maxval=1., dtype=tf.float32)
        self._embedding = tf.random_uniform(
            [self._vocab_size, self._emb_dim], maxval=1., dtype=tf.float32)
        self._encoder_output = tf.random_uniform(
            [self._batch_size, self._max_time, 64])

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
                "kwargs": {"num_units": self._attention_dim}
            }
        }
        decoder = tx.modules.AttentionRNNDecoder(
            vocab_size=self._vocab_size,
            memory=self._encoder_output,
            memory_sequence_length=encoder_values_length,
            hparams=hparams)

        outputs, final_state = beam_search_decode(
            decoder_or_cell=decoder,
            embedding=self._embedding,
            start_tokens=[1]*self._batch_size,
            end_token=2,
            beam_width=1,
            max_decoding_length=20)

        self.assertIsInstance(
            outputs, tf.contrib.seq2seq.FinalBeamSearchDecoderOutput)
        self.assertIsInstance(
            final_state, tf.contrib.seq2seq.BeamSearchDecoderState)

        beam_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
            cell=decoder.cell,
            embedding=self._embedding,
            start_tokens=[1]*self._batch_size,
            end_token=2,
            initial_state=decoder.cell.zero_state(self._batch_size, tf.float32),
            beam_width=1,
            output_layer=decoder.output_layer)

        outputs_1, final_state_1, _ = dynamic_decode(
            decoder=beam_decoder, maximum_iterations=20)

        outputs_2, _ = beam_search_decode(
            decoder_or_cell=decoder,
            embedding=self._embedding,
            start_tokens=[1]*self._batch_size,
            end_token=2,
            beam_width=11,
            max_decoding_length=21)
        outputs_3, _ = beam_search_decode(
            decoder_or_cell=decoder,
            embedding=self._embedding,
            start_tokens=[1]*self._batch_size,
            end_token=2,
            beam_width=11,
            max_decoding_length=21,
            output_time_major=True)

        with self.test_session() as sess:
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

if __name__ == "__main__":
    tf.test.main()
