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
from texar.modules.embedders.embedders import WordEmbedder

# pylint: disable=too-many-locals

class UnidirectionalRNNEncoderTest(tf.test.TestCase):
    """Tests :class:`~texar.modules.UnidirectionalRNNEncoder` class.
    """

    def test_trainable_variables(self):
        """Tests the functionality of automatically collecting trainable
        variables.
        """
        inputs = tf.placeholder(dtype=tf.float32, shape=[None, None, 100])

        # case 1
        encoder = UnidirectionalRNNEncoder()
        _, _ = encoder(inputs)
        self.assertEqual(len(encoder.trainable_variables), 2)

        # case 2
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

        # case 3
        hparams = {
            "output_layer": {
                "num_layers": 2,
                "layer_size": [100, 6],
                "activation": "relu",
                "final_layer_activation": "identity",
                "dropout_layer_ids": [0, 1, 2],
                "variational_dropout": False
            }
        }
        encoder = UnidirectionalRNNEncoder(hparams=hparams)
        _, _ = encoder(inputs)
        self.assertEqual(len(encoder.trainable_variables), 2+2+2)
        _, _ = encoder(inputs)
        self.assertEqual(len(encoder.trainable_variables), 2+2+2)

    def test_encode(self):
        """Tests encoding.
        """
        # case 1
        encoder = UnidirectionalRNNEncoder()

        max_time = 8
        batch_size = 16
        emb_dim = 100
        inputs = tf.random_uniform([batch_size, max_time, emb_dim],
                                   maxval=1., dtype=tf.float32)
        outputs, state = encoder(inputs)

        cell_dim = encoder.hparams.rnn_cell.kwargs.num_units
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs_, state_ = sess.run([outputs, state])
            self.assertEqual(outputs_.shape, (batch_size, max_time, cell_dim))
            self.assertEqual(state_[0].shape, (batch_size, cell_dim))

        # case 2: with output layers
        hparams = {
            "output_layer": {
                "num_layers": 2,
                "layer_size": [100, 6],
                "dropout_layer_ids": [0, 1, 2],
                "variational_dropout": True
            }
        }
        encoder = UnidirectionalRNNEncoder(hparams=hparams)

        max_time = 8
        batch_size = 16
        emb_dim = 100
        inputs = tf.random_uniform([batch_size, max_time, emb_dim],
                                   maxval=1., dtype=tf.float32)
        outputs, state, cell_outputs, output_size = encoder(
            inputs, return_cell_output=True, return_output_size=True)

        self.assertEqual(output_size[0], 6)
        self.assertEqual(cell_outputs.shape[-1], encoder.cell.output_size)

        out_dim = encoder.hparams.output_layer.layer_size[-1]
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs_ = sess.run(outputs)
            self.assertEqual(outputs_.shape, (batch_size, max_time, out_dim))


    def test_encode_with_embedder(self):
        """Tests encoding companioned with :mod:`texar.modules.embedders`.
        """
        embedder = WordEmbedder(vocab_size=20, hparams={"dim": 100})
        inputs = tf.ones([64, 16], dtype=tf.int32)

        encoder = UnidirectionalRNNEncoder()
        outputs, state = encoder(embedder(inputs))

        cell_dim = encoder.hparams.rnn_cell.kwargs.num_units
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
        inputs = tf.placeholder(dtype=tf.float32, shape=[None, None, 100])

        # case 1
        encoder = BidirectionalRNNEncoder()
        _, _ = encoder(inputs)
        self.assertEqual(len(encoder.trainable_variables), 4)

        # case 2
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

        # case 3
        hparams = {
            "output_layer_fw": {
                "num_layers": 2,
                "layer_size": [100, 6],
                "activation": "relu",
                "final_layer_activation": "identity",
                "dropout_layer_ids": [0, 1, 2],
                "variational_dropout": False
            },
            "output_layer_bw": {
                "num_layers": 3,
                "other_dense_kwargs": {"use_bias": False}
            },
            "output_layer_share_config": False
        }
        encoder = BidirectionalRNNEncoder(hparams=hparams)
        _, _ = encoder(inputs)
        self.assertEqual(len(encoder.trainable_variables), 4+4+3)
        _, _ = encoder(inputs)
        self.assertEqual(len(encoder.trainable_variables), 4+4+3)

    def test_encode(self):
        """Tests encoding.
        """
        # case 1
        encoder = BidirectionalRNNEncoder()

        max_time = 8
        batch_size = 16
        emb_dim = 100
        inputs = tf.random_uniform([batch_size, max_time, emb_dim],
                                   maxval=1., dtype=tf.float32)
        outputs, state = encoder(inputs)

        cell_dim = encoder.hparams.rnn_cell_fw.kwargs.num_units
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs_, state_ = sess.run([outputs, state])
            self.assertEqual(outputs_[0].shape,
                             (batch_size, max_time, cell_dim))
            self.assertEqual(state_[0][0].shape, (batch_size, cell_dim))

        # case 2: with output layers
        hparams = {
            "output_layer_fw": {
                "num_layers": 2,
                "layer_size": [100, 6],
                "dropout_layer_ids": [0, 1, 2],
                "variational_dropout": True
            }
        }
        encoder = BidirectionalRNNEncoder(hparams=hparams)

        max_time = 8
        batch_size = 16
        emb_dim = 100
        inputs = tf.random_uniform([batch_size, max_time, emb_dim],
                                   maxval=1., dtype=tf.float32)
        outputs, state, cell_outputs, output_size = encoder(
            inputs, return_cell_output=True, return_output_size=True)

        self.assertEqual(output_size[0][0], 6)
        self.assertEqual(output_size[1][0], 6)
        self.assertEqual(cell_outputs[0].shape[-1], encoder.cell_fw.output_size)
        self.assertEqual(cell_outputs[1].shape[-1], encoder.cell_bw.output_size)

        out_dim = encoder.hparams.output_layer_fw.layer_size[-1]
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs_ = sess.run(outputs)
            self.assertEqual(outputs_[0].shape, (batch_size, max_time, out_dim))
            self.assertEqual(outputs_[1].shape, (batch_size, max_time, out_dim))

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
#        cell_dim = encoder.hparams.rnn_cell.kwargs.num_units
#        with self.test_session() as sess:
#            sess.run(tf.global_variables_initializer())
#            outputs_, state_ = sess.run([outputs, state])
#            self.assertEqual(outputs_.shape, (batch_size, max_major_time, cell_dim))
#            self.assertEqual(state_[0].shape, (batch_size, cell_dim))

if __name__ == "__main__":
    tf.test.main()
