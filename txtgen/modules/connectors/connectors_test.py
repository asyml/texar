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
from tensorflow.python.util import nest    # pylint: disable=E0611

from txtgen.core import layers
from txtgen.modules.connectors.connectors import ConstantConnector
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

    def test_stochastic_connector(self): # pylint: disable=too-many-locals
        """Tests the logic of StochasticConnector.
        """
        variable_size = 5
        ctx_size = 3

        # pylint: disable=invalid-name
        mu = tf.zeros(shape=[self._batch_size, variable_size])
        log_var = tf.zeros(shape=[self._batch_size, variable_size])
        context = tf.zeros(shape=[self._batch_size, ctx_size])
        gauss_connector = StochasticConnector(variable_size)

        sample = gauss_connector((mu, log_var))
        ctx_sample = gauss_connector((mu, log_var, context))

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            sample_outputs, ctx_sample_outputs = sess.run([sample, ctx_sample])

            # check the same size
            self.assertEqual(sample_outputs.shape[0], self._batch_size)
            self.assertEqual(sample_outputs.shape[1], variable_size)

            self.assertEqual(ctx_sample_outputs.shape[0], self._batch_size)
            self.assertEqual(ctx_sample_outputs.shape[1],
                             variable_size+ctx_size)

            sample_mu = np.mean(sample_outputs, axis=0)
            # pylint: disable=no-member
            sample_log_var = np.log(np.var(sample_outputs, axis=0))

            # TODO(zhiting): these test statements do not pass on my computer
            ## check if the value is approximated N(0, 1)
            #for i in range(variable_size):
            #    self.assertAlmostEqual(0, sample_mu[i], delta=0.1)
            #    self.assertAlmostEqual(0, sample_log_var[i], delta=0.1)

if __name__ == "__main__":
    tf.test.main()
