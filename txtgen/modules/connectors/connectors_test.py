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
from txtgen.modules.connectors.connectors import ReparameterizeStochasticConnector
from txtgen.modules.connectors.connectors import StochasticConnector


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

    def test_reparameterize_stochastic_connector(self): # pylint: disable=too-many-locals
        """Tests the logic of RepaStochasticConnector.
        """
        variable_size = 5
        ctx_size = 3

        # pylint: disable=invalid-name
        mu = tf.zeros(shape=[self._batch_size, variable_size])
        var = tf.ones(shape=[self._batch_size, variable_size])
        gauss_connector = ReparameterizeStochasticConnector(variable_size)
        gauss_ds = tfds.MultivariateNormalDiag(loc = mu, scale_diag = var)


        sample = gauss_connector(gauss_ds)

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

if __name__ == "__main__":
    tf.test.main()
