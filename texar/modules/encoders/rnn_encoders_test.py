#
"""
Unit tests for RNN encoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf

from texar.modules.encoders.rnn_encoders import UnidirectionalRNNEncoder
from texar.modules.encoders.rnn_encoders import BidirectionalRNNEncoder
#from texar.modules.encoders.rnn_encoders import HierarchicalForwardRNNEncoder


class UnidirectionalRNNEncoderTest(tf.test.TestCase):
    """Tests :class:`~texar.modules.UnidirectionalRNNEncoder` class.
    """

    def test_trainable_variables(self):
        """Tests the functionality of automatically collecting trainable
        variables.
        """
        encoder = UnidirectionalRNNEncoder(vocab_size=2)

        inputs = [[1, 0]]
        _, _ = encoder(inputs)
        self.assertEqual(len(encoder.trainable_variables), 3)

        hparams = {
            "rnn_cell": {
                "dropout": {
                    "input_keep_prob": 0.5
                }
            }
        }
        encoder = UnidirectionalRNNEncoder(vocab_size=2, hparams=hparams)
        inputs = [[1, 0]]
        _, _ = encoder(inputs)
        self.assertEqual(len(encoder.trainable_variables), 3)

    def test_encode(self):
        """Tests encoding.
        """
        vocab_size = 4
        encoder = UnidirectionalRNNEncoder(vocab_size=vocab_size)

        max_time = 8
        batch_size = 16
        inputs = tf.random_uniform([batch_size, max_time],
                                   maxval=vocab_size,
                                   dtype=tf.int32)
        outputs, state = encoder(inputs)

        cell_dim = encoder.hparams.rnn_cell.cell.kwargs.num_units
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs_, state_ = sess.run([outputs, state])
            self.assertEqual(outputs_.shape, (batch_size, max_time, cell_dim))
            self.assertEqual(state_[0].shape, (batch_size, cell_dim))

class BidirectionalRNNEncoderTest(tf.test.TestCase):
    """Tests :class:`~texar.modules.BidirectionalRNNEncoder` class.
    """

    def test_trainable_variables(self):
        """Tests the functionality of automatically collecting trainable
        variables.
        """
        encoder = BidirectionalRNNEncoder(vocab_size=2)

        inputs = [[1, 0]]
        _, _ = encoder(inputs)
        self.assertEqual(len(encoder.trainable_variables), 5)

        hparams = {
            "rnn_cell_fw": {
                "dropout": {
                    "input_keep_prob": 0.5
                }
            }
        }
        encoder = BidirectionalRNNEncoder(vocab_size=2, hparams=hparams)
        inputs = [[1, 0]]
        _, _ = encoder(inputs)
        self.assertEqual(len(encoder.trainable_variables), 5)

    def test_encode(self):
        """Tests encoding.
        """
        vocab_size = 4
        encoder = BidirectionalRNNEncoder(vocab_size=vocab_size)

        max_time = 8
        batch_size = 16
        inputs = tf.random_uniform([batch_size, max_time],
                                   maxval=vocab_size,
                                   dtype=tf.int32)
        outputs, state = encoder(inputs)

        cell_dim = encoder.hparams.rnn_cell_fw.cell.kwargs.num_units
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs_, state_ = sess.run([outputs, state])
            self.assertEqual(outputs_[0].shape,
                             (batch_size, max_time, cell_dim))
            self.assertEqual(state_[0][0].shape, (batch_size, cell_dim))


# TODO(zhiting): not completed yet
#class HierarchicalForwardRNNEncoderTest(tf.test.TestCase):
#    """Tests HierarchicalForwardRNNEncoder class.
#    """
#
#    def test_trainable_variables(self):
#        """Tests the functionality of automatically collecting trainable
#        variables.
#        """
#        encoder = HierarchicalForwardRNNEncoder(vocab_size=2)
#        inputs = [[[1, 0], [0, 1], [0, 1]]]
#        _, _ = encoder(inputs)
#        self.assertEqual(len(encoder.trainable_variables), 5)
#
#    def test_encode(self):
#        """Tests encoding.
#        """
#        vocab_size = 4
#        encoder = HierarchicalForwardRNNEncoder(vocab_size=vocab_size)
#
#        max_major_time = 8
#        max_minor_time = 6
#        batch_size = 16
#        inputs = tf.random_uniform([batch_size, max_major_time, max_minor_time],
#                                   maxval=vocab_size,
#                                   dtype=tf.int32)
#        outputs, state = encoder(inputs)
#
#        cell_dim = encoder.hparams.rnn_cell.cell.kwargs.num_units
#        with self.test_session() as sess:
#            sess.run(tf.global_variables_initializer())
#            outputs_, state_ = sess.run([outputs, state])
#            self.assertEqual(outputs_.shape, (batch_size, max_major_time, cell_dim))
#            self.assertEqual(state_[0].shape, (batch_size, cell_dim))

if __name__ == "__main__":
    tf.test.main()
