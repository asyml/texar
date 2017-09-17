#
"""
Various connectors
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.util import nest    # pylint: disable=E0611

from txtgen.modules.connectors.connector_base import ConnectorBase
from txtgen.core.utils import get_function
from txtgen.core import distributions


def _mlp_transform(inputs, output_size, activation_fn=tf.identity):
    """Transforms inputs through a fully-connected layer that creates the output
    with specified size.

    Args:
        inputs: A Tensor of shape `[batch_size, ...]` (i.e., batch-major), or a
            (nested) tuple of such elements. A Tensor or a (nested) tuple of
            Tensors with shape `[max_time, batch_size, ...]` (i.e., time-major)
            can be transposed to batch-major using
            `txtgen.core.utils.transpose_batch_time` prior to this function.
        output_size: Can be an Integer, a TensorShape, or a (nested) tuple of
            Integers or TensorShape.
        activation_fn: Activation function applied to the output.

    Returns:
        If `output_size` is an Integer or a TensorShape, returns a Tensor of
        shape `[batch_size x output_size]`. If `output_size` is a tuple of
        Integers or TensorShape, returns a tuple having the same structure as
        `output_size`, where each element Tensor has the same size as defined
        in `output_size`.
    """
    # flatten inputs
    flat_input = nest.flatten(inputs)
    batch_size = flat_input[0].shape[0].value
    flat_input = [tf.reshape(input_, [batch_size, -1]) for input_ in flat_input]
    concat_input = tf.concat(flat_input, 1)

    # get output dimension
    flat_output_size = nest.flatten(output_size)
    sum_output_size = sum(flat_output_size)

    fc_output = tf.contrib.layers.fully_connected(
        concat_input, sum_output_size, activation_fn=activation_fn)

    flat_output = tf.split(fc_output, flat_output_size, axis=1)
    output = nest.pack_sequence_as(structure=output_size,
                                   flat_sequence=flat_output)

    return output


class ForwardConnector(ConnectorBase):
    """Simply forward the final state of encoder to decoder without
    transformation.

    The encoder and decoder must have the same state structures and sizes.
    """

    def __init__(self, decoder_state_size, name="forward_connector"):
        """Initializes the connector.

        Args:
            decoder_state_size: Size of state of the decoder cell. Can be an
                Integer, a Tensorshape , or a tuple of Integers or TensorShapes.
                This can typically be obtained by `decoder.cell.state_size`.
            name: Name of connector.
        """
        ConnectorBase.__init__(self, decoder_state_size, None, name)

    def _build(self, encoder_state):    # pylint: disable=W0221
        """Passes the encoder outputs to the initial states of decoder.

        Args:
            encoder_state: The state of encoder to pass forward. Must have the
                same sizes with the decoder state.

        Returns:
            The state of the encoder.
        """
        nest.assert_same_structure(encoder_state,
                                   self._decoder_state_size)
        self._built = True

        return encoder_state

    @staticmethod
    def default_hparams():
        return {}


class MLPTransformConnector(ConnectorBase):
    """Transforms the encoder results (e.g., outputs or final states) with an
    MLP layer. Takes the outputs as the decoder initial state.
    """

    def __init__(self, decoder_state_size, hparams=None, name="mlp_connector"):
        ConnectorBase.__init__(self, decoder_state_size, hparams, name)

    def _build(self, encoder_result): #pylint: disable=W0221
        """Transforms the encoder results with an MLP layer.

        Args:
            encoder_result: Result of encoder (e.g., encoder outputs or final
                states) to be transformed and passed to the decoder. Must be a
                Tensor of shape `[batch_size, ...]` or a (nested) tuple of such
                Tensors.

        Returns:
            A Tensor or a (nested) tuple of Tensors of the same structure of
            the decoder state.
        """
        fn_modules = ['txtgen.custom', 'tensorflow', 'tensorflow.nn']
        activation_fn = get_function(self.hparams.activation_fn, fn_modules)

        decoder_state = _mlp_transform(
            encoder_result, self._decoder_state_size, activation_fn)

        self._add_internal_trainable_variables()
        self._built = True

        return decoder_state

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        Returns:
            ```python
            {
                # The name or full path of the activation function applied to
                # the outputs of the MLP layer. E.g., the name of built-in
                # functions defined in module `tensorflow` or `tensorflow.nn`,
                # or user-defined functions defined in `user.custom`, or a
                # full path like "my_module.my_activation_fn".

                "activation_fn": "tensorflow.identity"
            }
            ```
        """
        return {
            "activation_fn": "tensorflow.identity"
        }


class StochasticConnector(ConnectorBase):
    #    """Samples decoder initial state from a distribution defined by the
    #    encoder outputs.
    #
    #    Used in, e.g., variational autoencoders, adversarial autoencoders, and other
    #    models.
    #    """

    def __init__(self, decoder_state_size, hparams=None, name="stochastic_connector"):
        ConnectorBase.__init__(self, decoder_state_size, hparams, name)

    def _build(self, encoder_result):  # pylint: disable=W0221
        """Transforms the encoder results with an MLP layer.

        Args:
            encoder_result: Result of encoder (e.g., encoder outputs or final
                states) to be transformed and passed to the decoder. Must be a
                Tensor of shape `[batch_size, ...]` or a (nested) tuple of such
                Tensors.
                
                gaussian_distribution only supports type encoder results. The 
                encoder results can either be (mu, logvar) or (mu, logvar, context) with 
                size [batch_size, ...] for tuple element. When context is present, it will be
                concatenated with the sample along the 1st axis.

        Returns:
            A Tensor or a (nested) tuple of Tensors of the same structure of
            the decoder state.
        """
        sampler = get_function(self.hparams.distribution)

        if sampler is distributions.sample_gaussian:
            if type(encoder_result) is not tuple:
                raise ValueError("Gaussian connector requires tuple encoder results")

            if len(encoder_result) == 2:
                encoder_mu, encoder_log_var = encoder_result
                decoder_state = sampler(encoder_mu, encoder_log_var)

            elif len(encoder_result) == 3:
                encoder_mu, encoder_log_var, context = encoder_result
                sample = sampler(encoder_mu, encoder_log_var)
                decoder_state = tf.concat([sample, context], axis=1)
            else:
                raise ValueError("Gaussian connector supports either (mu, logvar) or (mu, logvar, context)")

        else:
            raise ValueError("Unsupported distribution")

        self._add_internal_trainable_variables()
        self._built = True

        return decoder_state

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        Returns:
            ```python
            {
                # The name or full path of the activation function applied to
                # the outputs of the MLP layer. E.g., the name of built-in
                # functions defined in module `tensorflow` or `tensorflow.nn`,
                # or user-defined functions defined in `user.custom`, or a
                # full path like "my_module.my_activation_fn".

                "activation_fn": "tensorflow.identity"
            }
            ```
        """
        return {
            "distribution": "txtgen.core.distributions.sample_gaussian"
        }

