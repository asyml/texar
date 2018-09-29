#
"""
Unit tests for various layers.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import tensorflow as tf
import tensorflow.contrib.rnn as rnn

import texar as tx
from texar import context
from texar.hyperparams import HParams
from texar.core import layers

# pylint: disable=no-member, protected-access, invalid-name
# pylint: disable=redefined-variable-type

class GetRNNCellTest(tf.test.TestCase):
    """Tests RNN cell creator.
    """

    def test_get_rnn_cell(self):
        """Tests :func:`texar.core.layers.get_rnn_cell`.
        """
        emb_dim = 4
        num_units = 64

        # Given instance
        hparams = {
            "type": rnn.LSTMCell(num_units)
        }
        cell = layers.get_rnn_cell(hparams)
        self.assertTrue(isinstance(cell, rnn.LSTMCell))

        # Given class
        hparams = {
            "type": rnn.LSTMCell,
            "kwargs": {"num_units": 10}
        }
        cell = layers.get_rnn_cell(hparams)
        self.assertTrue(isinstance(cell, rnn.LSTMCell))

        # Given string, and complex hyperparameters
        keep_prob_x = tf.placeholder(
            name='keep_prob', shape=[], dtype=tf.float32)
        hparams = {
            "type": "tensorflow.contrib.rnn.GRUCell",
            "kwargs": {
                "num_units": num_units
            },
            "num_layers": 2,
            "dropout": {
                "input_keep_prob": 0.8,
                "state_keep_prob": keep_prob_x,
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

            feed_dict = {
                keep_prob_x: 1.0,
                context.global_mode(): tf.estimator.ModeKeys.TRAIN
            }
            output_, state_ = sess.run([output, state], feed_dict=feed_dict)

            self.assertEqual(output_.shape[0], batch_size)
            if isinstance(state_, (list, tuple)):
                self.assertEqual(state_[0].shape[0], batch_size)
                self.assertEqual(state_[0].shape[1],
                                 hparams_.kwargs.num_units)
            else:
                self.assertEqual(state_.shape[0], batch_size)
                self.assertEqual(state_.shape[1],
                                 hparams_.kwargs.num_units)


    def test_switch_dropout(self):
        """Tests dropout mode.
        """
        emb_dim = 4
        num_units = 64
        hparams = {
            "kwargs": {
                "num_units": num_units
            },
            "num_layers": 2,
            "dropout": {
                "input_keep_prob": 0.8,
            },
        }
        mode = tf.placeholder(tf.string)
        hparams_ = HParams(hparams, layers.default_rnn_cell_hparams())
        cell = layers.get_rnn_cell(hparams_, mode)

        batch_size = 16
        inputs = tf.zeros([batch_size, emb_dim], dtype=tf.float32)
        output, state = cell(inputs,
                             cell.zero_state(batch_size, dtype=tf.float32))
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            output_train, _ = sess.run(
                [output, state],
                feed_dict={mode: tf.estimator.ModeKeys.TRAIN})
            self.assertEqual(output_train.shape[0], batch_size)
            output_test, _ = sess.run(
                [output, state],
                feed_dict={mode: tf.estimator.ModeKeys.EVAL})
            self.assertEqual(output_test.shape[0], batch_size)


class GetActivationFnTest(tf.test.TestCase):
    """Tests :func:`texar.core.layers.get_activation_fn`.
    """
    def test_get_activation_fn(self):
        """Tests.
        """
        fn = layers.get_activation_fn()
        self.assertEqual(fn, tf.identity)

        fn = layers.get_activation_fn('relu')
        self.assertEqual(fn, tf.nn.relu)

        inputs = tf.random_uniform([64, 100], -5, 20, dtype=tf.int32)

        fn = layers.get_activation_fn('leaky_relu')
        fn_output = fn(inputs)
        ref_output = tf.nn.leaky_relu(inputs)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            fn_output_, ref_output_ = sess.run([fn_output, ref_output])
            np.testing.assert_array_equal(fn_output_, ref_output_)

        fn = layers.get_activation_fn('leaky_relu', kwargs={'alpha': 0.1})
        fn_output = fn(inputs)
        ref_output = tf.nn.leaky_relu(inputs, alpha=0.1)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            fn_output_, ref_output_ = sess.run([fn_output, ref_output])
            np.testing.assert_array_equal(fn_output_, ref_output_)


class GetLayerTest(tf.test.TestCase):
    """Tests layer creator.
    """
    def test_get_layer(self):
        """Tests :func:`texar.core.layers.get_layer`.
        """
        hparams = {
            "type": "Conv1D"
        }
        layer = layers.get_layer(hparams)
        self.assertTrue(isinstance(layer, tf.layers.Conv1D))

        hparams = {
            "type": "MergeLayer",
            "kwargs": {
                "layers": [
                    {"type": "Conv1D"},
                    {"type": "Conv1D"}
                ]
            }
        }
        layer = layers.get_layer(hparams)
        self.assertTrue(isinstance(layer, tx.core.MergeLayer))

        hparams = {
            "type": tf.layers.Conv1D
        }
        layer = layers.get_layer(hparams)
        self.assertTrue(isinstance(layer, tf.layers.Conv1D))

        hparams = {
            "type": tf.layers.Conv1D(filters=10, kernel_size=2)
        }
        layer = layers.get_layer(hparams)
        self.assertTrue(isinstance(layer, tf.layers.Conv1D))


class ReducePoolingLayerTest(tf.test.TestCase):
    """Tests reduce pooling layer.
    """
    def setUp(self):
        tf.test.TestCase.setUp(self)

        self._batch_size = 64
        self._seq_length = 16
        self._emb_dim = 100

    def test_max_reduce_pooling_layer(self):
        """Tests :class:`texar.core.MaxReducePooling1D`.
        """
        pool_layer = layers.MaxReducePooling1D()

        inputs = tf.random_uniform(
            [self._batch_size, self._seq_length, self._emb_dim])
        output_shape = pool_layer.compute_output_shape(inputs.get_shape())
        output = pool_layer(inputs)
        output_reduce = tf.reduce_max(inputs, axis=1)
        self.assertEqual(output.get_shape(), output_shape)
        self.assertEqual(output.get_shape(), [self._batch_size, self._emb_dim])

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            output_, output_reduce_ = sess.run([output, output_reduce])
            np.testing.assert_array_equal(output_, output_reduce_)

    def test_average_reduce_pooling_layer(self):
        """Tests :class:`texar.core.AverageReducePooling1D`.
        """
        pool_layer = layers.AverageReducePooling1D()

        inputs = tf.random_uniform(
            [self._batch_size, self._seq_length, self._emb_dim])
        output_shape = pool_layer.compute_output_shape(inputs.get_shape())
        output = pool_layer(inputs)
        output_reduce = tf.reduce_mean(inputs, axis=1)
        self.assertEqual(output.get_shape(), output_shape)
        self.assertEqual(output.get_shape(), [self._batch_size, self._emb_dim])

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            output_, output_reduce_ = sess.run([output, output_reduce])
            np.testing.assert_array_equal(output_, output_reduce_)

class MergeLayerTest(tf.test.TestCase):
    """Tests MergeLayer.
    """

    def test_output_shape(self):
        """Tests MergeLayer.compute_output_shape function.
        """
        input_shapes = [[None, 1, 2], [64, 2, 2], [None, 3, 2]]

        concat_layer = layers.MergeLayer(mode='concat', axis=1)
        concat_output_shape = concat_layer.compute_output_shape(input_shapes)
        self.assertEqual(concat_output_shape, [64, 6, 2])

        sum_layer = layers.MergeLayer(mode='sum', axis=1)
        sum_output_shape = sum_layer.compute_output_shape(input_shapes)
        self.assertEqual(sum_output_shape, [64, 2])

        input_shapes = [[None, 5, 2], [64, None, 2], [2]]
        esum_layer = layers.MergeLayer(mode='elemwise_sum')
        esum_output_shape = esum_layer.compute_output_shape(input_shapes)
        self.assertEqual(esum_output_shape, [64, 5, 2])

    def test_layer_logics(self):
        """Test the logic of MergeLayer.
        """
        layers_ = []
        layers_.append(tf.layers.Conv1D(filters=200, kernel_size=3))
        layers_.append(tf.layers.Conv1D(filters=200, kernel_size=4))
        layers_.append(tf.layers.Conv1D(filters=200, kernel_size=5))
        layers_.append(tf.layers.Dense(200))
        layers_.append(tf.layers.Dense(200))
        m_layer = layers.MergeLayer(layers_)

        inputs = tf.zeros([64, 16, 1024], dtype=tf.float32)
        outputs = m_layer(inputs)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs_ = sess.run(outputs)
            self.assertEqual(outputs_.shape[0], 64)
            self.assertEqual(outputs_.shape[2], 200)
            self.assertEqual(
                outputs_.shape,
                m_layer.compute_output_shape(inputs.shape.as_list()))

    def test_trainable_variables(self):
        """Test the trainable_variables of the layer.
        """
        layers_ = []
        layers_.append(tf.layers.Conv1D(filters=200, kernel_size=3))
        layers_.append(tf.layers.Conv1D(filters=200, kernel_size=4))
        layers_.append(tf.layers.Conv1D(filters=200, kernel_size=5))
        layers_.append(tf.layers.Dense(200))
        layers_.append(tf.layers.Dense(200))
        m_layer = layers.MergeLayer(layers_)

        inputs = tf.zeros([64, 16, 1024], dtype=tf.float32)
        _ = m_layer(inputs)

        num_vars = sum([len(layer.trainable_variables) for layer in layers_])
        self.assertEqual(num_vars, len(m_layer.trainable_variables))

class SequentialLayerTest(tf.test.TestCase):
    """Tests sequential layer.
    """

    def test_seq_layer(self):
        """Test sequential layer.
        """
        layers_ = []
        layers_.append(tf.layers.Dense(100))
        layers_.append(tf.layers.Dense(200))
        seq_layer = layers.SequentialLayer(layers_)

        output_shape = seq_layer.compute_output_shape([None, 10])
        self.assertEqual(output_shape[1].value, 200)

        inputs = tf.zeros([10, 20], dtype=tf.float32)
        outputs = seq_layer(inputs)

        num_vars = sum([len(layer.trainable_variables) for layer in layers_])
        self.assertEqual(num_vars, len(seq_layer.trainable_variables))

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs_ = sess.run(outputs)
            self.assertEqual(outputs_.shape[0], 10)
            self.assertEqual(outputs_.shape[1], 200)


if __name__ == "__main__":
    tf.test.main()
