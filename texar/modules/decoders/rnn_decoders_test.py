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

# pylint: disable=no-member, too-many-locals, too-many-instance-attributes
# pylint: disable=too-many-arguments, protected-access


class BasicRNNDecoderTest(tf.test.TestCase):
    """Tests :class:`~texar.modules.decoders.rnn_decoders.BasicRNNDecoder`.
    """

    def setUp(self):
        tf.test.TestCase.setUp(self)
        self._vocab_size = 4
        self._max_time = 8
        self._batch_size = 16
        self._emb_dim = 20
        self._inputs = tf.random_uniform(
            [self._batch_size, self._max_time, self._emb_dim],
            maxval=1., dtype=tf.float32)
        self._embedding = tf.random_uniform(
            [self._vocab_size, self._emb_dim], maxval=1., dtype=tf.float32)

    def _test_outputs(self, decoder, outputs, final_state, sequence_lengths,
                      test_mode=False):
        # 4 trainable variables: cell-kernel, cell-bias,
        # fc-layer-weights, fc-layer-bias
        self.assertEqual(len(decoder.trainable_variables), 4)

        cell_dim = decoder.hparams.rnn_cell.kwargs.num_units
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            outputs_, final_state_, sequence_lengths_ = sess.run(
                [outputs, final_state, sequence_lengths],
                feed_dict={context.global_mode(): tf.estimator.ModeKeys.TRAIN})
            self.assertIsInstance(outputs_, BasicRNNDecoderOutput)
            if not test_mode:
                self.assertEqual(
                    outputs_.logits.shape,
                    (self._batch_size, self._max_time, self._vocab_size))
                self.assertEqual(
                    outputs_.sample_id.shape,
                    (self._batch_size, self._max_time))
                np.testing.assert_array_equal(
                    sequence_lengths_, [self._max_time]*self._batch_size)
            self.assertEqual(final_state_[0].shape,
                             (self._batch_size, cell_dim))

    def test_output_layer(self):
        decoder = BasicRNNDecoder(vocab_size=self._vocab_size,
                                  output_layer=None)
        self.assertIsInstance(decoder, BasicRNNDecoder)

        decoder = BasicRNNDecoder(output_layer=tf.identity)
        self.assertIsInstance(decoder, BasicRNNDecoder)

        tensor = tf.random_uniform(
            [self._emb_dim, self._vocab_size], maxval=1, dtype=tf.float32
        )
        decoder = BasicRNNDecoder(output_layer=tensor)
        self.assertIsInstance(decoder, BasicRNNDecoder)
        self.assertEqual(decoder.vocab_size, self._vocab_size)


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
            **decoder.hparams.helper_train.kwargs.todict())
        outputs, final_state, sequence_lengths = decoder(helper=helper_train)
        self._test_outputs(decoder, outputs, final_state, sequence_lengths)

        outputs, final_state, sequence_lengths = decoder(
            inputs=self._inputs,
            sequence_length=[self._max_time]*self._batch_size)
        self._test_outputs(decoder, outputs, final_state, sequence_lengths)

        outputs, final_state, sequence_lengths = decoder(
            decoding_strategy=None,
            inputs=self._inputs,
            sequence_length=[self._max_time]*self._batch_size)
        self._test_outputs(decoder, outputs, final_state, sequence_lengths)

        outputs, final_state, sequence_lengths = decoder(
            decoding_strategy=None,
            embedding=self._embedding,
            start_tokens=[1]*self._batch_size,
            end_token=2,
            mode=tf.estimator.ModeKeys.EVAL)
        self._test_outputs(decoder, outputs, final_state, sequence_lengths,
                           test_mode=True)

    def test_decode_train_with_tf(self):
        """Compares decoding results with TF built-in decoder.
        """
        _inputs_placeholder = tf.placeholder(
            tf.int32, [self._batch_size, self._max_time], name="inputs")
        _embedding_placeholder = tf.placeholder(
            tf.float32, [self._vocab_size, self._emb_dim], name="emb")
        inputs = tf.nn.embedding_lookup(_embedding_placeholder,
                                        _inputs_placeholder)

        output_layer = tf.layers.Dense(self._vocab_size)
        decoder = BasicRNNDecoder(vocab_size=self._vocab_size,
                                  output_layer=output_layer)

        helper_train = get_helper(
            decoder.hparams.helper_train.type,
            inputs=inputs,
            sequence_length=[self._max_time]*self._batch_size,
            **decoder.hparams.helper_train.kwargs.todict())

        outputs, final_state, sequence_lengths = decoder(helper=helper_train)

        tf_helper = tf.contrib.seq2seq.TrainingHelper(
            inputs, [self._max_time]*self._batch_size)

        tf_decoder = tf.contrib.seq2seq.BasicDecoder(
            decoder.cell,
            tf_helper,
            decoder.cell.zero_state(self._batch_size, tf.float32),
            output_layer=output_layer)

        tf_outputs, tf_final_state, tf_sequence_lengths = \
            tf.contrib.seq2seq.dynamic_decode(tf_decoder)

        cell_dim = decoder.hparams.rnn_cell.kwargs.num_units
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            inputs_ = np.random.randint(
                self._vocab_size, size=(self._batch_size, self._max_time),
                dtype=np.int32)
            embedding_ = np.random.randn(self._vocab_size, self._emb_dim)

            outputs_, final_state_, sequence_lengths_ = sess.run(
                [outputs, final_state, sequence_lengths],
                feed_dict={context.global_mode(): tf.estimator.ModeKeys.TRAIN,
                           _inputs_placeholder: inputs_,
                           _embedding_placeholder: embedding_})
            self.assertEqual(final_state_[0].shape,
                             (self._batch_size, cell_dim))

            tf_outputs_, tf_final_state_, tf_sequence_lengths_ = sess.run(
                [tf_outputs, tf_final_state, tf_sequence_lengths],
                feed_dict={context.global_mode(): tf.estimator.ModeKeys.TRAIN,
                           _inputs_placeholder: inputs_,
                           _embedding_placeholder: embedding_})

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
            embedding=self._embedding,
            start_tokens=[self._vocab_size-2]*self._batch_size,
            end_token=self._vocab_size-1,
            **decoder.hparams.helper_train.kwargs.todict())

        outputs, final_state, sequence_lengths = decoder(helper=helper_infer)

        # 4 trainable variables: embedding, cell-kernel, cell-bias,
        # fc-layer-weights, fc-layer-bias
        self.assertEqual(len(decoder.trainable_variables), 4)

        cell_dim = decoder.hparams.rnn_cell.kwargs.num_units
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs_, final_state_, sequence_lengths_ = sess.run(
                [outputs, final_state, sequence_lengths],
                feed_dict={context.global_mode():
                           tf.estimator.ModeKeys.PREDICT})
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
        self._emb_dim = 20
        self._attention_dim = 256
        self._inputs = tf.random_uniform(
            [self._batch_size, self._max_time, self._emb_dim],
            maxval=1., dtype=tf.float32)
        self._embedding = tf.random_uniform(
            [self._vocab_size, self._emb_dim], maxval=1., dtype=tf.float32)
        self._encoder_output = tf.random_uniform(
            [self._batch_size, self._max_time, 64])

    def test_decode_train(self):
        """Tests decoding in training mode.
        """
        seq_length = np.random.randint(
            self._max_time, size=[self._batch_size]) + 1
        encoder_values_length = tf.constant(seq_length)
        hparams = {
            "attention": {
                "kwargs": {
                    "num_units": self._attention_dim,
                    # Note: to use sparsemax in TF-CPU, it looks
                    # `memory_sequence_length` must equal max_time.
                    #"probability_fn": "sparsemax"
                }
            }
        }
        decoder = AttentionRNNDecoder(
            memory=self._encoder_output,
            memory_sequence_length=encoder_values_length,
            vocab_size=self._vocab_size,
            hparams=hparams)

        helper_train = get_helper(
            decoder.hparams.helper_train.type,
            inputs=self._inputs,
            sequence_length=[self._max_time]*self._batch_size,
            **decoder.hparams.helper_train.kwargs.todict())

        outputs, final_state, sequence_lengths = decoder(helper=helper_train)
        # 4+1 trainable variables: cell-kernel, cell-bias,
        # fc-weight, fc-bias, and
        # memory_layer: For LuongAttention, we only transform the memory layer;
        # thus num_units *must* match the expected query depth.
        self.assertEqual(len(decoder.trainable_variables), 5)

        cell_dim = decoder.hparams.rnn_cell.kwargs.num_units
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs_, final_state_, sequence_lengths_ = sess.run(
                [outputs, final_state, sequence_lengths],
                feed_dict={context.global_mode(): tf.estimator.ModeKeys.TRAIN})
            self.assertIsInstance(outputs_, AttentionRNNDecoderOutput)
            self.assertEqual(
                outputs_.logits.shape,
                (self._batch_size, self._max_time, self._vocab_size))
            self.assertEqual(
                outputs_.sample_id.shape, (self._batch_size, self._max_time))
            self.assertEqual(final_state_.cell_state[0].shape,
                             (self._batch_size, cell_dim))
            np.testing.assert_array_equal(
                sequence_lengths_, [self._max_time]*self._batch_size)


    def test_decode_infer(self):
        """Tests decoding in inference mode.
        """
        seq_length = np.random.randint(
            self._max_time, size=[self._batch_size]) + 1
        encoder_values_length = tf.constant(seq_length)
        hparams = {
            "attention": {
                "kwargs": {
                    "num_units": 256,
                }
            }
        }
        decoder = AttentionRNNDecoder(
            vocab_size=self._vocab_size,
            memory=self._encoder_output,
            memory_sequence_length=encoder_values_length,
            hparams=hparams)

        helper_infer = get_helper(
            decoder.hparams.helper_infer.type,
            embedding=self._embedding,
            start_tokens=[1]*self._batch_size,
            end_token=2,
            **decoder.hparams.helper_train.kwargs.todict())

        outputs, final_state, sequence_lengths = decoder(helper=helper_infer)

        # 4+1 trainable variables: cell-kernel, cell-bias,
        # fc-weight, fc-bias, and
        # memory_layer: For LuongAttention, we only transform the memory layer;
        # thus num_units *must* match the expected query depth.
        self.assertEqual(len(decoder.trainable_variables), 5)
        cell_dim = decoder.hparams.rnn_cell.kwargs.num_units
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs_, final_state_, sequence_lengths_ = sess.run(
                [outputs, final_state, sequence_lengths],
                feed_dict={context.global_mode():
                           tf.estimator.ModeKeys.PREDICT})
            self.assertIsInstance(outputs_, AttentionRNNDecoderOutput)
            max_length = max(sequence_lengths_)
            self.assertEqual(
                outputs_.logits.shape,
                (self._batch_size, max_length, self._vocab_size))
            self.assertEqual(
                outputs_.sample_id.shape, (self._batch_size, max_length))
            self.assertEqual(final_state_.cell_state[0].shape,
                             (self._batch_size, cell_dim))

    def test_beam_search_cell(self):
        """Tests :meth:`texar.modules.AttentionRNNDecoder._get_beam_search_cell`
        """
        seq_length = np.random.randint(
            self._max_time, size=[self._batch_size]) + 1
        encoder_values_length = tf.constant(seq_length)
        hparams = {
            "attention": {
                "kwargs": {
                    "num_units": self._attention_dim,
                    "probability_fn": "sparsemax"
                }
            }
        }
        decoder = AttentionRNNDecoder(
            memory=self._encoder_output,
            memory_sequence_length=encoder_values_length,
            vocab_size=self._vocab_size,
            hparams=hparams)

        helper_train = get_helper(
            decoder.hparams.helper_train.type,
            inputs=self._inputs,
            sequence_length=[self._max_time]*self._batch_size,
            **decoder.hparams.helper_train.kwargs.todict())

        _, _, _ = decoder(helper=helper_train)

        ## 4+1 trainable variables: cell-kernel, cell-bias,
        ## fc-weight, fc-bias, and
        ## memory_layer: For LuongAttention, we only transform the memory layer;
        ## thus num_units *must* match the expected query depth.
        self.assertEqual(len(decoder.trainable_variables), 5)

        beam_width = 3
        beam_cell = decoder._get_beam_search_cell(beam_width)
        cell_input = tf.random_uniform([self._batch_size * beam_width,
                                        self._emb_dim])
        cell_state = beam_cell.zero_state(self._batch_size * beam_width,
                                          tf.float32)
        _ = beam_cell(cell_input, cell_state)
        # Test if beam_cell is sharing variables with decoder cell.
        for tvar in beam_cell.trainable_variables:
            self.assertTrue(tvar in decoder.trainable_variables)

if __name__ == "__main__":
    tf.test.main()
