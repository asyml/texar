#
"""
Unit tests for RNN encoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf

from texar.modules.encoders.hierarchical_encoders import HierarchicalRNNEncoder

# pylint: disable=too-many-locals

class HierarchicalRNNEncoderTest(tf.test.TestCase):
    """Tests HierarchicalRNNEncoder
    """

    def test_trainable_variables(self):
        encoder = HierarchicalRNNEncoder()

        inputs = tf.random_uniform(
            [3, 2, 3, 4],
            maxval=1,
            minval=-1,
            dtype=tf.float32)
        _, _ = encoder(inputs)

        self.assertEqual(
            len(encoder.trainable_variables),
            len(encoder.encoder_major.trainable_variables) + \
            len(encoder.encoder_minor.trainable_variables))

    def test_encode(self):
        encoder = HierarchicalRNNEncoder()

        batch_size = 16
        max_major_time = 8
        max_minor_time = 6
        dim = 10
        inputs = tf.random_uniform(
            [batch_size, max_major_time, max_minor_time, dim],
            maxval=1,
            minval=-1,
            dtype=tf.float32)
        outputs, state = encoder(inputs)

        cell_dim = encoder.encoder_major.hparams.rnn_cell.kwargs.num_units

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs_, state_ = sess.run([outputs, state])
            self.assertEqual(state_[0].shape, (batch_size, cell_dim))

    def test_order(self):
        encoder = HierarchicalRNNEncoder()

        batch_size = 16
        max_major_time = 8
        max_minor_time = 6
        dim = 10
        inputs = tf.random_uniform(
            [batch_size, max_major_time, max_minor_time, dim],
            maxval=1,
            minval=-1,
            dtype=tf.float32)

        outputs_1, state_1 = encoder(inputs, order='btu')
        outputs_2, state_2 = encoder(inputs, order='utb')
        outputs_3, state_3 = encoder(inputs, order='tbu')
        outputs_4, state_4 = encoder(inputs, order='ubt')

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run([outputs_1, state_1, outputs_2, state_2,
                      outputs_3, state_3, outputs_4, state_4])

    def test_depack(self):
        hparams = {
            "encoder_major_type": "BidirectionalRNNEncoder",
            "encoder_major_hparams": {
                "rnn_cell_fw": {
                    "type": "LSTMCell",
                    "kwargs": {
                        "num_units": 100
                    }
                }
            }
        }
        encoder = HierarchicalRNNEncoder(hparams=hparams)

        batch_size = 16
        max_major_time = 8
        max_minor_time = 6
        dim = 10
        inputs = tf.random_uniform(
            [batch_size, max_major_time, max_minor_time, dim],
            maxval=1,
            minval=-1,
            dtype=tf.float32)

        _, _ = encoder(inputs)

        self.assertEqual(
            encoder.states_minor_before_medium.h.shape[1],
            encoder.states_minor_after_medium.shape[1])

    def test_encoder_minor_as_birnn(self):
        """Tests encoder_minor as a BidirectionalRNNEncoder
        """
        hparams = {
            "encoder_minor_type": "BidirectionalRNNEncoder",
            "encoder_minor_hparams": {
                "rnn_cell_fw": {
                    "type": "LSTMCell",
                    "kwargs": {
                        "num_units": 100
                    }
                }
            },
            "encoder_major_hparams": {
                "rnn_cell": {
                    "type": "LSTMCell",
                    "kwargs": {
                        "num_units": 200
                    }
                }
            }
        }
        encoder = HierarchicalRNNEncoder(hparams=hparams)

        batch_size = 16
        max_major_time = 8
        max_minor_time = 6
        dim = 10
        inputs = tf.random_uniform(
            [batch_size, max_major_time, max_minor_time, dim],
            maxval=1,
            minval=-1,
            dtype=tf.float32)

        outputs, _ = encoder(inputs)
        self.assertEqual(list(outputs.shape), [16, 8, 200])

if __name__ == "__main__":
    tf.test.main()
