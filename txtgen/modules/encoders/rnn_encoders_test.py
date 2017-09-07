#
"""
Unit tests for RNN encoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf

from txtgen.modules.encoders.rnn_encoders import ForwardRNNEncoder


class ForwardENNEncoderTest(tf.test.TestCase):
    """Tests ForwardRNNEncoder class.
    """

    def test_trainable_variables(self):
        """Tests the functionality of automatically collecting trainable
        variables.
        """
        encoder = ForwardRNNEncoder(vocab_size=2)
        self.assertEqual(len(encoder.trainable_variables), 0)

        inputs = [[1, 0]]
        _, _ = encoder(inputs)
        self.assertEqual(len(encoder.trainable_variables), 3)

    def test_encode(self):
        """Tests encoding.
        """
        vocab_size = 4
        encoder = ForwardRNNEncoder(vocab_size=vocab_size)

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

if __name__ == "__main__":
    tf.test.main()
