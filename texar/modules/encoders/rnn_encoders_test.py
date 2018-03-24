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
from texar.modules.encoders import HierarchicalRNNEncoder
from texar.modules.embedders.embedders import WordEmbedder

class UnidirectionalRNNEncoderTest(tf.test.TestCase):
    """Tests :class:`~texar.modules.UnidirectionalRNNEncoder` class.
    """

    def test_trainable_variables(self):
        """Tests the functionality of automatically collecting trainable
        variables.
        """
        inputs = tf.ones([64, 16, 100])

        encoder = UnidirectionalRNNEncoder()
        _, _ = encoder(inputs)
        self.assertEqual(len(encoder.trainable_variables), 2)

        hparams = {
            "rnn_cell": {
                "dropout": {
                    "input_keep_prob": 0.5
                }
            }
        }
        encoder = UnidirectionalRNNEncoder(hparams=hparams)
        _, _ = encoder(inputs)
        self.assertEqual(len(encoder.trainable_variables), 2)

    def test_encode(self):
        """Tests encoding.
        """
        encoder = UnidirectionalRNNEncoder()

        max_time = 8
        batch_size = 16
        emb_dim = 100
        inputs = tf.random_uniform([batch_size, max_time, emb_dim],
                                   maxval=1., dtype=tf.float32)
        outputs, state = encoder(inputs)

        cell_dim = encoder.hparams.rnn_cell.cell.kwargs.num_units
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs_, state_ = sess.run([outputs, state])
            self.assertEqual(outputs_.shape, (batch_size, max_time, cell_dim))
            self.assertEqual(state_[0].shape, (batch_size, cell_dim))

    def test_encode_with_embedder(self):
        """Tests encoding companioned with :mod:`texar.modules.embedders`.
        """
        embedder = WordEmbedder(vocab_size=20, hparams={"dim": 100})
        inputs = tf.ones([64, 16], dtype=tf.int32)

        encoder = UnidirectionalRNNEncoder()
        outputs, state = encoder(embedder(inputs))

        cell_dim = encoder.hparams.rnn_cell.cell.kwargs.num_units
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs_, state_ = sess.run([outputs, state])
            self.assertEqual(outputs_.shape, (64, 16, cell_dim))
            self.assertEqual(state_[0].shape, (64, cell_dim))

class BidirectionalRNNEncoderTest(tf.test.TestCase):
    """Tests :class:`~texar.modules.BidirectionalRNNEncoder` class.
    """

    def test_trainable_variables(self):
        """Tests the functionality of automatically collecting trainable
        variables.
        """
        inputs = tf.ones([64, 16, 100])

        encoder = BidirectionalRNNEncoder()
        _, _ = encoder(inputs)
        self.assertEqual(len(encoder.trainable_variables), 4)

        hparams = {
            "rnn_cell_fw": {
                "dropout": {
                    "input_keep_prob": 0.5
                }
            }
        }
        encoder = BidirectionalRNNEncoder(hparams=hparams)
        _, _ = encoder(inputs)
        self.assertEqual(len(encoder.trainable_variables), 4)

    def test_encode(self):
        """Tests encoding.
        """
        encoder = BidirectionalRNNEncoder()

        max_time = 8
        batch_size = 16
        emb_dim = 100
        inputs = tf.random_uniform([batch_size, max_time, emb_dim],
                                   maxval=1., dtype=tf.float32)
        outputs, state = encoder(inputs)

        cell_dim = encoder.hparams.rnn_cell_fw.cell.kwargs.num_units
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs_, state_ = sess.run([outputs, state])
            self.assertEqual(outputs_[0].shape,
                             (batch_size, max_time, cell_dim))
            self.assertEqual(state_[0][0].shape, (batch_size, cell_dim))


class HierarchicalForwardRNNEncoderTest(tf.test.TestCase):
    """Tests HierarchicalForwardRNNEncoder class.
    """

    def test_trainable_variables(self):
        """Tests the functionality of automatically collecting trainable
        variables.
        """
        encoder = HierarchicalRNNEncoder()
        inputs = tf.constant([[[[1, 0], [0, 1], [0., 1]]]])
        _, _ = encoder(inputs)
        self.assertEqual(len(encoder.trainable_variables), 4)

    def test_encode(self):
        """Tests encoding.
        """
        vocab_size = 4
        encoder = HierarchicalRNNEncoder()

        max_major_time = 8
        max_minor_time = 6
        batch_size = 16
        dims = 32
        inputs = tf.random_uniform([batch_size, max_major_time, max_minor_time, dims],
                                   maxval=0.1,
                                   minval=-0.1,
                                   dtype=tf.float32)
        outputs, state = encoder(inputs)

        cell_dim = encoder.encoder_minor.hparams.rnn_cell.cell.kwargs.num_units
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs_, state_ = sess.run([outputs, state])
            self.assertEqual(outputs_.shape, (batch_size, max_major_time, cell_dim))
            self.assertEqual(state_[0].shape, (batch_size, cell_dim))

    
    def test_bidir_encode(self):
        """Tests encoding.
        """
        vocab_size = 4

        minor_encoder = BidirectionalRNNEncoder()
        encoder = HierarchicalRNNEncoder(encoder_minor=minor_encoder)

        max_major_time = 8
        max_minor_time = 6
        batch_size = 16
        dims = 32
        inputs = tf.random_uniform([batch_size, max_major_time, max_minor_time, dims],
                                   maxval=0.1,
                                   minval=-0.1,
                                   dtype=tf.float32)
        outputs, state = encoder(inputs)

        cell_dim = encoder.encoder_major.hparams.rnn_cell.cell.kwargs.num_units
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs_, state_ = sess.run([outputs, state])
            self.assertEqual(outputs_.shape, (batch_size, max_major_time, cell_dim))
            self.assertEqual(state_[0].shape, (batch_size, cell_dim))

if __name__ == "__main__":
    tf.test.main()
