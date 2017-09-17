# -*- coding: utf-8 -*-
# author: Tiancheng Zhao

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
from txtgen.modules.connectors.connectors import StochasticConnector
import numpy as np


class TestStochasticConnector(tf.test.TestCase):
    def test__build(self):
        batch_size = 1000
        variable_size = 5
        mu = tf.zeros(shape=[batch_size, variable_size])
        log_var = tf.zeros(shape=[batch_size, variable_size])
        gauss_connector = StochasticConnector(variable_size)
        sample = gauss_connector((mu, log_var))

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            sample_outputs = sess.run(sample)

            # check the same size
            self.assertEqual(sample_outputs.shape[0], batch_size)
            self.assertEqual(sample_outputs.shape[1], variable_size)

            sample_mu = np.mean(sample_outputs, axis=0)
            sample_log_var = np.log(np.var(sample_outputs, axis=0))

            # check if the value is approximated N(0, 1)
            for i in range(variable_size):
                self.assertAlmostEqual(0, sample_mu[i], delta=0.1)
                self.assertAlmostEqual(0, sample_log_var[i], delta=0.1)

tf.test.main()