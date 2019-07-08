#
"""
Unit tests for connectors.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
from tensorflow_probability import distributions as tfpd
from tensorflow.python.util import nest    # pylint: disable=E0611

from texar.core import layers
from texar.modules import ConstantConnector
from texar.modules import MLPTransformConnector
from texar.modules import (ReparameterizedStochasticConnector,
                           StochasticConnector)
from texar.modules.connectors.connectors import _assert_same_size

# pylint: disable=too-many-locals, invalid-name

class TestConnectors(tf.test.TestCase):
    """Tests various connectors.
    """

    def setUp(self):
        tf.test.TestCase.setUp(self)
        self._batch_size = 100

        self._decoder_cell = layers.get_rnn_cell(
            layers.default_rnn_cell_hparams())

    def test_constant_connector(self):
        """Tests the logic of
        :class:`~texar.modules.connectors.ConstantConnector`.
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
        """Tests the logic of
        :class:`~texar.modules.connectors.ForwardConnector`.
        """
        # TODO(zhiting)
        pass

    def test_mlp_transform_connector(self):
        """Tests the logic of
        :class:`~texar.modules.connectors.MLPTransformConnector`.
        """
        connector = MLPTransformConnector(self._decoder_cell.state_size)
        output = connector(tf.zeros([5, 10]))
        nest.assert_same_structure(output, self._decoder_cell.state_size)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            output_ = sess.run(output)
            nest.assert_same_structure(output_, self._decoder_cell.state_size)

    def test_reparameterized_stochastic_connector(self):
        """Tests the logic of
        :class:`~texar.modules.ReparameterizedStochasticConnector`.
        """
        state_size = (10, 10)
        variable_size = 100
        state_size_ts = (tf.TensorShape([10, 10]), tf.TensorShape([2, 3, 4]))
        sample_num = 10

        mu = tf.zeros([self._batch_size, variable_size])
        var = tf.ones([self._batch_size, variable_size])
        mu_vec = tf.zeros([variable_size])
        var_vec = tf.ones([variable_size])
        gauss_ds = tfpd.MultivariateNormalDiag(loc=mu, scale_diag=var)
        gauss_ds_vec = tfpd.MultivariateNormalDiag(loc=mu_vec,
                                                   scale_diag=var_vec)
        gauss_connector = ReparameterizedStochasticConnector(state_size)
        gauss_connector_ts = ReparameterizedStochasticConnector(state_size_ts)

        output_1, _ = gauss_connector(gauss_ds)
        output_2, _ = gauss_connector(
            distribution="MultivariateNormalDiag",
            distribution_kwargs={"loc": mu, "scale_diag": var})
        sample_ts, _ = gauss_connector_ts(gauss_ds)

        # specify sample num
        sample_test_num, _ = gauss_connector(
            gauss_ds_vec, num_samples=sample_num)

        # test when :attr:`transform` is False
        #sample_test_no_transform = gauss_connector(gauss_ds, transform=False)

        test_list = [output_1, output_2, sample_ts, sample_test_num]

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out_list = sess.run(test_list)
            out1 = out_list[0]
            out2 = out_list[1]
            out_ts = out_list[2]
            out_test_num = out_list[3]

            # check the same size
            self.assertEqual(out_test_num[0].shape,
                             tf.TensorShape([sample_num, state_size[0]]))
            self.assertEqual(out1[0].shape,
                             tf.TensorShape([self._batch_size, state_size[0]]))
            self.assertEqual(out2[0].shape,
                             tf.TensorShape([self._batch_size, state_size[0]]))
            _assert_same_size(out_ts, state_size_ts)

            # sample_mu = np.mean(sample_outputs, axis=0)
            # # pylint: disable=no-member
            # sample_var = np.var(sample_outputs, axis=0)

            ## check if the value is approximated N(0, 1)
            # for i in range(variable_size):
               # self.assertAlmostEqual(0, sample_mu[i], delta=0.2)
               # self.assertAlmostEqual(1, sample_var[i], delta=0.2)

    def test_stochastic_connector(self):
        """Tests the logic of
        :class:`~texar.modules.StochasticConnector`.
        """
        state_size = (10, 10)
        variable_size = 100
        state_size_ts = tf.TensorShape([self._batch_size, variable_size])
        gauss_connector = StochasticConnector(state_size)
        mu = tf.zeros([self._batch_size, variable_size])
        var = tf.ones([self._batch_size, variable_size])
        gauss_ds = tfpd.MultivariateNormalDiag(loc=mu, scale_diag=var)
        output_1, _ = gauss_connector(gauss_ds)

        gauss_connector_2 = StochasticConnector(state_size_ts)
        output_2, sample2 = gauss_connector_2(
            distribution="MultivariateNormalDiag",
            distribution_kwargs={"loc": mu, "scale_diag": var}, transform=False)
        test_list = [output_1, output_2, sample2]

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out_list = sess.run(test_list)
            out1 = out_list[0]
            out2 = out_list[1]
            sample2 = out_list[2]
            self.assertEqual(out1[0].shape,
                             tf.TensorShape([self._batch_size, state_size[0]]))
            self.assertEqual(out2.shape, state_size_ts)
            self.assertEqual(out2.shape, sample2.shape)

    #def test_concat_connector(self): # pylint: disable=too-many-locals
    #    """Tests the logic of
    #    :class:`~texar.modules.connectors.ConcatConnector`.
    #    """
    #    gauss_size = 5
    #    constant_size = 7
    #    variable_size = 13

    #    decoder_size1 = 16
    #    decoder_size2 = (16, 32)

    #    gauss_connector = StochasticConnector(gauss_size)
    #    categorical_connector = StochasticConnector(1)
    #    constant_connector = ConstantConnector(constant_size)
    #    concat_connector1 = ConcatConnector(decoder_size1)
    #    concat_connector2 = ConcatConnector(decoder_size2)

    #    # pylint: disable=invalid-name
    #    mu = tf.zeros([self._batch_size, gauss_size])
    #    var = tf.ones([self._batch_size, gauss_size])
    #    categorical_prob = tf.constant(
    #       [[0.1, 0.2, 0.7] for _ in xrange(self._batch_size)])
    #    categorical_ds = tfds.Categorical(probs = categorical_prob)
    #    gauss_ds = tfds.MultivariateNormalDiag(loc = mu, scale_diag = var)

    #    gauss_state = gauss_connector(gauss_ds)
    #    categorical_state = categorical_connector(categorical_ds)
    #    constant_state = constant_connector(self._batch_size, value=1.)
    #    with tf.Session() as debug_sess:
    #        debug_cater = debug_sess.run(categorical_state)

    #    state1 = concat_connector1(
    #       [gauss_state, categorical_state, constant_state])
    #    state2 = concat_connector2(
    #       [gauss_state, categorical_state, constant_state])

    #    with self.test_session() as sess:
    #        sess.run(tf.global_variables_initializer())
    #        [output1, output2] = sess.run([state1, state2])

    #        # check the same size
    #        self.assertEqual(output1.shape[1], decoder_size1)
    #        self.assertEqual(output2[1].shape[1], decoder_size2[1])

if __name__ == "__main__":
    tf.test.main()
