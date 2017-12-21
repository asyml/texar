#
"""
Unit tests for various layers.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import tensorflow as tf
import tensorflow.contrib.rnn as rnn

from texar import context
from texar.hyperparams import HParams
from texar.core import layers

# pylint: disable=no-member

class GetRNNCellTest(tf.test.TestCase):
    """Tests RNN cell creator.
    """

    def test_get_rnn_cell(self):
        """Tests :func:`texar.core.layers.get_rnn_cell`.
        """
        emb_dim = 4
        num_units = 64

        hparams = {
            "cell": {
                "type": rnn.BasicLSTMCell(num_units)
            }
        }
        cell = layers.get_rnn_cell(hparams)
        self.assertTrue(isinstance(cell, rnn.BasicLSTMCell))

        hparams = {
            "cell": {
                "type": "tensorflow.contrib.rnn.GRUCell",
                "kwargs": {
                    "num_units": num_units
                }
            },
            "num_layers": 2,
            "dropout": {
                "input_keep_prob": 0.8,
                "variational_recurrent": True,
                "input_size": [emb_dim, num_units]
            },
            "residual": True,
            "highway": True
        }

        hparams_ = HParams(hparams, layers.default_rnn_cell_hparams())
        cell = layers.get_rnn_cell(hparams_)

        batch_size = 16
        inputs = tf.zeros([batch_size, emb_dim], dtype=tf.float32)
        output, state = cell(inputs,
                             cell.zero_state(batch_size, dtype=tf.float32))
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            output_, state_ = sess.run([output, state],
                                       feed_dict={context.is_train(): True})
            self.assertEqual(output_.shape[0], batch_size)
            if isinstance(state_, (list, tuple)):
                self.assertEqual(state_[0].shape[0], batch_size)
                self.assertEqual(state_[0].shape[1],
                                 hparams_.cell.kwargs.num_units)
            else:
                self.assertEqual(state_.shape[0], batch_size)
                self.assertEqual(state_.shape[1],
                                 hparams_.cell.kwargs.num_units)

    def test_get_embedding(self):
        """Tests :func:`texar.core.layers.get_embedding`.
        """
        vocab_size = 100
        emb = layers.get_embedding(vocab_size=vocab_size)
        self.assertEqual(emb.shape[0].value, vocab_size)
        self.assertEqual(emb.shape[1].value,
                         layers.default_embedding_hparams()["dim"])

        hparams = {
            "name": "embedding_2",
            "initializer": {
                "type": tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
            },
            "regularizer": {
                "type": tf.keras.regularizers.L1L2(0.1, 0.1)
            }
        }
        emb = layers.get_embedding(hparams=hparams, vocab_size=vocab_size)
        self.assertEqual(emb.shape[0].value, vocab_size)
        self.assertEqual(emb.shape[1].value,
                         layers.default_embedding_hparams()["dim"])

    def test_get_layer(self):
        """Tests :func:`texar.core.layers.get_layer`.
        """
        hparams = {
            "type": "Conv1D"
        }
        layer = layers.get_layer(hparams)
        self.assertTrue(isinstance(layer, tf.layers.Conv1D))

class SequentialLayerTest(tf.test.TestCase):
    """Test sequential layer.
    """

    def test_seq_layer(self):
        #TODO(zichao): add docstring
        layers_ = []
        layers_.append(tf.layers.Dense(100))
        layers_.append(tf.layers.Dense(200))
        seq_layer = layers.SequentialLayer(layers_)
        output_shape = seq_layer._compute_output_shape([None, 10]) # pylint: disable=protected-access
        self.assertEqual(output_shape[1].value, 200)

        inputs = tf.zeros([10, 20], dtype=tf.float32)
        outputs = seq_layer(inputs)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs_ = sess.run(outputs)
            self.assertEqual(outputs_.shape[0], 10)
            self.assertEqual(outputs_.shape[1], 200)

if __name__ == "__main__":
    tf.test.main()
