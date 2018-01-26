"""
Unit tests for RNN decoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import tensorflow as tf

from texar.modules.decoders.rnn_decoders import BasicRNNDecoderOutput
from texar.modules.decoders.rnn_decoders import BasicRNNDecoder
from texar.modules.decoders.rnn_decoders import AttentionRNNDecoderOutput
from texar.modules.decoders.rnn_decoders import AttentionRNNDecoder
from texar.modules.decoders.rnn_decoder_helpers import get_helper
from texar import context

# pylint: disable=no-member

class BasicRNNDecoderTest(tf.test.TestCase):
    """Tests :class:`~texar.modules.decoders.rnn_decoders.BasicRNNDecoder`.
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
        output_layer = tf.layers.Dense(self._vocab_size)
        decoder = BasicRNNDecoder(vocab_size=self._vocab_size,
                                  output_layer=output_layer)

        helper_train = get_helper(
            decoder.hparams.helper_train.type,
            inputs=self._inputs,
            sequence_length=[self._max_time]*self._batch_size,
            embedding=decoder.embedding,
            **decoder.hparams.helper_train.kwargs.todict())

        outputs, final_state, sequence_lengths = decoder(
            helper_train, decoder.cell.zero_state(self._batch_size, tf.float32))

        # 5 trainable variables: embedding, cell-kernel, cell-bias,
        # fc-layer-weights, fc-layer-bias
        print(decoder.trainable_variables)
        self.assertEqual(len(decoder.trainable_variables), 5)

        cell_dim = decoder.hparams.rnn_cell.cell.kwargs.num_units
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            outputs_, final_state_, sequence_lengths_ = sess.run(
                [outputs, final_state, sequence_lengths],
                feed_dict={context.is_train(): True})
            self.assertIsInstance(outputs_, BasicRNNDecoderOutput)
            self.assertEqual(
                outputs_.logits.shape,
                (self._batch_size, self._max_time, self._vocab_size))
            self.assertEqual(
                outputs_.sample_id.shape, (self._batch_size, self._max_time))
            self.assertEqual(final_state_[0].shape,
                             (self._batch_size, cell_dim))
            np.testing.assert_array_equal(
                sequence_lengths_,
                [self._max_time]*self._batch_size)

    def test_decode_train_with_tf(self):
        self._inputs_placeholder = tf.placeholder(
            tf.int32, [self._batch_size, self._max_time], name="inputs")
        output_layer = tf.layers.Dense(self._vocab_size)
        decoder = BasicRNNDecoder(vocab_size=self._vocab_size,
                                  output_layer=output_layer)

        helper_train = get_helper(
            decoder.hparams.helper_train.type,
            inputs=self._inputs_placeholder,
            sequence_length=[self._max_time]*self._batch_size,
            embedding=decoder.embedding,
            **decoder.hparams.helper_train.kwargs.todict())

        outputs, final_state, sequence_lengths = decoder(
            helper_train, decoder.cell.zero_state(self._batch_size, tf.float32))

        inputs = tf.nn.embedding_lookup(decoder.embedding,
                                        self._inputs_placeholder)
        tf_helper = tf.contrib.seq2seq.TrainingHelper(
            inputs, [self._max_time]*self._batch_size)

        tf_decoder = tf.contrib.seq2seq.BasicDecoder(
            decoder.cell,
            tf_helper,
            decoder.cell.zero_state(self._batch_size, tf.float32),
            output_layer=output_layer)

        tf_outputs, tf_final_state, tf_sequence_lengths \
            = tf.contrib.seq2seq.dynamic_decode(
                tf_decoder)

        cell_dim = decoder.hparams.rnn_cell.cell.kwargs.num_units
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            inputs_ = np.random.randint(
                self._vocab_size, size=(self._batch_size, self._max_time),
                dtype=np.int32)

            outputs_, final_state_, sequence_lengths_ = sess.run(
                [outputs, final_state, sequence_lengths],
                feed_dict={context.is_train(): True,
                           self._inputs_placeholder: inputs_})
            self.assertEqual(final_state_[0].shape,
                             (self._batch_size, cell_dim))

            tf_outputs_, tf_final_state_, tf_sequence_lengths_ = sess.run(
                [tf_outputs, tf_final_state, tf_sequence_lengths],
                feed_dict={context.is_train(): True,
                           self._inputs_placeholder: inputs_})

            np.testing.assert_array_equal(outputs_.logits,
                                          tf_outputs_.rnn_output)
            np.testing.assert_array_equal(outputs_.sample_id,
                                          tf_outputs_.sample_id)
            np.testing.assert_array_equal(final_state_.c, tf_final_state_.c)
            np.testing.assert_array_equal(final_state_.h, tf_final_state_.h)
            np.testing.assert_array_equal(sequence_lengths_,
                                          tf_sequence_lengths_)

    def test_decode_infer(self):
        """Tests decoding in inferencee mode.
        """
        output_layer = tf.layers.Dense(self._vocab_size)
        decoder = BasicRNNDecoder(vocab_size=self._vocab_size,
                                  output_layer=output_layer)

        helper_infer = get_helper(
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
                feed_dict={context.is_train(): False})
            self.assertIsInstance(outputs_, BasicRNNDecoderOutput)
            max_length = max(sequence_lengths_)
            self.assertEqual(
                outputs_.logits.shape,
                (self._batch_size, max_length, self._vocab_size))
            self.assertEqual(
                outputs_.sample_id.shape, (self._batch_size, max_length))
            self.assertEqual(final_state_[0].shape,
                             (self._batch_size, cell_dim))


class AttentionRNNDecoderTest(tf.test.TestCase):
    """Tests :class:`~texar.modules.decoders.rnn_decoders.AttentionRNNDecoder`.
    """

    def setUp(self):
        tf.test.TestCase.setUp(self)
        self._vocab_size = 10
        self._max_time = 16
        self._batch_size = 8
        self._attention_dim = 64
        self._inputs = tf.random_uniform([self._batch_size, self._max_time],
                                         maxval=self._vocab_size,
                                         dtype=tf.int32)
        self._encoder_output = tf.random_uniform(
            [self._batch_size, self._max_time, 64])

    def test_decode_train(self):
        """Tests decoding in training mode.
        """
        seq_length = np.random.randint(
            self._max_time, size=[self._batch_size]) + 1
        encoder_values_length = tf.constant(seq_length)
        decoder = AttentionRNNDecoder(
            memory=self._encoder_output,
            memory_sequence_length=encoder_values_length,
            vocab_size=self._vocab_size,
            hparams=None)   # Use default hyperparameters

        helper_train = get_helper(
            decoder.hparams.helper_train.type,
            inputs=self._inputs,
            sequence_length=[self._max_time]*self._batch_size,
            embedding=decoder.embedding,
            **decoder.hparams.helper_train.kwargs.todict())

        outputs, final_state, sequence_lengths = decoder(
            helper_train, decoder.cell.zero_state(self._batch_size, tf.float32))

        cell_dim = decoder.hparams.rnn_cell.cell.kwargs.num_units
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs_, final_state_, sequence_lengths_ = sess.run(
                [outputs, final_state, sequence_lengths],
                feed_dict={context.is_train(): True})
            self.assertIsInstance(outputs_, AttentionRNNDecoderOutput)
            self.assertEqual(
                outputs_.logits.shape,
                (self._batch_size, self._max_time, self._vocab_size))
            self.assertEqual(
                outputs_.sample_id.shape, (self._batch_size, self._max_time))
            self.assertEqual(final_state_.cell_state[0].shape,
                             (self._batch_size, cell_dim))
            np.testing.assert_array_equal(
                sequence_lengths_,
                [self._max_time]*self._batch_size)


    def test_decode_infer(self):
        """Tests decoding in inference mode.
        """
        seq_length = np.random.randint(
            self._max_time, size=[self._batch_size]) + 1
        encoder_values_length = tf.constant(seq_length)
        decoder = AttentionRNNDecoder(
            vocab_size=self._vocab_size,
            memory=self._encoder_output,
            memory_sequence_length=encoder_values_length,
            hparams=None)

        helper_infer = get_helper(
            decoder.hparams.helper_infer.type,
            embedding=decoder.embedding,
            start_tokens=[1]*self._batch_size,
            end_token=2,
            **decoder.hparams.helper_train.kwargs.todict())

        outputs, final_state, sequence_lengths = decoder(
            helper_infer, decoder.cell.zero_state(self._batch_size, tf.float32))
        # 5+1 trainable variables: embedding, cell-kernel, cell-bias,
        # fc-weight, fc-bias, and
        # memory_layer: For LuongAttention, we only transform the memory layer;
        # thus num_units *must* match the expected query depth.
        self.assertEqual(len(decoder.trainable_variables), 6)
        cell_dim = decoder.hparams.rnn_cell.cell.kwargs.num_units
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs_, final_state_, sequence_lengths_ = sess.run(
                [outputs, final_state, sequence_lengths],
                feed_dict={context.is_train(): False})
            self.assertIsInstance(outputs_, AttentionRNNDecoderOutput)
            max_length = max(sequence_lengths_)
            self.assertEqual(
                outputs_.logits.shape,
                (self._batch_size, max_length, self._vocab_size))
            self.assertEqual(
                outputs_.sample_id.shape, (self._batch_size, max_length))
            self.assertEqual(final_state_.cell_state[0].shape,
                             (self._batch_size, cell_dim))

if __name__ == "__main__":
    tf.test.main()
