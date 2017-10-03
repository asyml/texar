# -*- coding: utf-8 -*-
# author: Tiancheng Zhao
"""
Unit tests for connectors.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import tensorflow as tf
import tensorflow.contrib.distributions as tfds
from tensorflow.python.util import nest    # pylint: disable=E0611

from txtgen.core import layers
from txtgen.modules.connectors.connectors import ConstantConnector
from txtgen.modules.connectors.connectors import ReparameterizedStochasticConnector
from txtgen.modules.connectors.connectors import StochasticConnector
from txtgen.modules.connectors.connectors import ConcatConnector


class TestConnectors(tf.test.TestCase):
    """Tests various connectors.
    """
    def setUp(self):
        tf.test.TestCase.setUp(self)
        self._batch_size = 100
        self._decoder_cell = layers.get_rnn_cell(
            layers.default_rnn_cell_hparams())

    def test_constant_connector(self):
        """Tests the logic of ConstantConnector.
        """
        connector = ConstantConnector(self._decoder_cell.state_size)
        decoder_initial_state_0 = connector(self._batch_size)
        decoder_initial_state_1 = connector(self._batch_size, value=1.)
        nest.assert_same_structure(decoder_initial_state_0,
                                   self._decoder_cell.state_size)
        nest.assert_same_structure(decoder_initial_state_1,
                                   self._decoder_cell.state_size)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            s_0, s_1 = sess.run(
                [decoder_initial_state_0, decoder_initial_state_1])
            self.assertEqual(nest.flatten(s_0)[0][0, 0], 0.)
            self.assertEqual(nest.flatten(s_1)[0][0, 0], 1.)

    def test_forward_connector(self):
        """Tests the logic of ForwardConnector.
        """
        # TODO(zhiting)
        pass

    def test_mlp_transform_connector(self):
        """Tests the logic of MLPTransformConnector.
        """
        # TODO(zhiting)
        pass

    def test_reparameterized_stochastic_connector(self): # pylint: disable=too-many-locals
        """Tests the logic of ReparameterizedStochasticConnector.
        """
        variable_size = 5

        # pylint: disable=invalid-name
        mu = tf.zeros(variable_size)
        var = tf.ones(variable_size)
        gauss_connector = ReparameterizedStochasticConnector(variable_size)
        gauss_ds = tfds.MultivariateNormalDiag(loc = mu, scale_diag = var)


        sample = gauss_connector(gauss_ds, self._batch_size)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            sample_outputs = sess.run(sample)

            # check the same size
            self.assertEqual(sample_outputs.shape[0], self._batch_size)
            self.assertEqual(sample_outputs.shape[1], variable_size)

            sample_mu = np.mean(sample_outputs, axis=0)
            # pylint: disable=no-member
            sample_var = np.var(sample_outputs, axis=0)

            ## check if the value is approximated N(0, 1)
            for i in range(variable_size):
               self.assertAlmostEqual(0, sample_mu[i], delta=0.2)
               self.assertAlmostEqual(1, sample_var[i], delta=0.2)

    def test_concat_connector(self): # pylint: disable=too-many-locals
        """Tests the logic of ConcatConnector.
        """
        gauss_size = 5
        constant_size = 7
        variable_size = 13

        categorical_connector = StochasticConnector(1)
        gauss_connector = ReparameterizedStochasticConnector(variable_size)
        constant_connector = ConstantConnector(constant_size)
        concat_connector = ConcatConnector(variable_size)


        # pylint: disable=invalid-name
        mu = tf.zeros(gauss_size)
        var = tf.ones(gauss_size)
        categorical_prob = [0.1, 0.2, 0.7]
        categorical_ds = tfds.Categorical(probs = categorical_prob)
        gauss_ds = tfds.MultivariateNormalDiag(loc = mu, scale_diag = var)
        print(gauss_connector(gauss_ds, self._batch_size).shape)
        print(categorical_connector(categorical_ds, self._batch_size).shape)
        print(constant_connector(self._batch_size, value=1.).shape)

        state = concat_connector([gauss_connector(gauss_ds, self._batch_size), categorical_connector(categorical_ds, self._batch_size), constant_connector(self._batch_size, value=1.)])

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            output = sess.run(state)

            # check the same size
            self.assertEqual(output.shape[0], self._batch_size)
            self.assertEqual(output.shape[1], variable_size)

if __name__ == "__main__":
    tf.test.main()
