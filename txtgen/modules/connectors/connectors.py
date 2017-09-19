#
"""
Various connectors.
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
            :meth:`~txtgen.core.utils.transpose_batch_time` prior to this
            function.
        output_size: Can be an Integer, a TensorShape, or a (nested) tuple of
            Integers or TensorShape.
        activation_fn: Activation function applied to the output.

    Returns:
        If :attr:`output_size` is an Integer or a TensorShape, returns a Tensor
        of shape `[batch_size x output_size]`. If :attr:`output_size` is a tuple
        of Integers or TensorShape, returns a tuple having the same structure as
        :attr:`output_size`, where each element Tensor has the same size as
        defined in :attr:`output_size`.
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

class ConstantConnector(ConnectorBase):
    """Creates decoder initial state that has a constant value.

    Args:
        decoder_state_size: Size of state of the decoder cell. Can be an
            Integer, a Tensorshape, or a tuple of Integers or TensorShapes.
            This can typically be obtained by :attr:`decoder.cell.state_size`.
        hparams (dict): Hyperparameters of the connector.
        name (str): Name of connector.
    """
    def __init__(self, decoder_state_size, hparams=None,
                 name="constant_connector"):
        ConnectorBase.__init__(self, decoder_state_size, hparams, name)

    @staticmethod
    def default_hparams():
        """Returns a dictionary of default hyperparameters:

        .. code-block:: python

            {
                # The constant value that the decoder initial state has.
                "value": 0.
            }
        """
        return {
            "value": 0.
        }

    def _build(self, batch_size, value=None):   # pylint: disable=W0221
        """Creates decoder initial state that has the given value.

        Args:
            batch_size (int or scalar int Tensor): The batch size.
            value (python number or scalar Tensor, optional): The value that
                the decoder initial state has. If `None` (default), the decoder
                initial state is set to :attr:`hparams.value`.

        Returns:
            A (structure of) tensor with the same structure of the decoder
            state, and with the given value.
        """
        value_ = value
        if value_ is None:
            value_ = self.hparams.value
        output = nest.map_structure(
            lambda x: tf.constant(value_, shape=[batch_size, x]),
            self._decoder_state_size)
        return output


class ForwardConnector(ConnectorBase):
    """Directly forwards input (structure of) tensors to decoder.

    The input must have the same structure with the decoder state,
    or must have the same number of elements and be re-packable into the decoder
    state structure. Note that if input is or contains a `dict` instance, the
    keys will be sorted to pack in deterministic order (See
    :meth:`~tensorflow.python.util.nest.pack_sequence_as` for more details).

    Args:
        decoder_state_size: Size of state of the decoder cell. Can be an
            Integer, a Tensorshape , or a tuple of Integers or TensorShapes.
            This can typically be obtained by :attr:`decoder.cell.state_size`.
        name (str): Name of connector.
    """

    def __init__(self, decoder_state_size, name="forward_connector"):
        ConnectorBase.__init__(self, decoder_state_size, None, name)

    @staticmethod
    def default_hparams():
        """Returns a dictionary of default hyperparameters.

        The dictionary is empty since the connector does not have any
        configurable hyperparameters.
        """
        return {}

    def _build(self, inputs):    # pylint: disable=W0221
        """Passes inputs to the initial states of decoder.

        :attr:`inputs` must either have the same structure, or the same number
        of elements with the decoder state.

        Args:
            inputs: The input (structure of) tensors to pass forward.

        Returns:
            The input (structure of) tensors that might be re-packed to have
            the same structure with decoder state.
        """
        output = inputs
        try:
            nest.assert_same_structure(inputs, self._decoder_state_size)
        except (ValueError, TypeError):
            flat_input = nest.flatten(inputs)
            output = nest.pack_sequence_as(
                self._decoder_state_size, flat_input)

        self._built = True

        return output


class MLPTransformConnector(ConnectorBase):
    """Transforms inputs with an MLP layer. Takes the outputs as the decoder
    initial state.

    Args:
        decoder_state_size: Size of state of the decoder cell. Can be an
            Integer, a Tensorshape , or a tuple of Integers or TensorShapes.
            This can typically be obtained by :attr:`decoder.cell.state_size`.
        hparams (dict): Hyperparameters of the connector.
        name (str): Name of connector.
    """

    def __init__(self, decoder_state_size, hparams=None, name="mlp_connector"):
        ConnectorBase.__init__(self, decoder_state_size, hparams, name)

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values:

        .. code-block:: python

            {
                # The name or full path of the activation function applied to
                # the outputs of the MLP layer. E.g., the name of built-in
                # functions defined in module `tensorflow` or `tensorflow.nn`,
                # or user-defined functions defined in `user.custom`, or a
                # full path like "my_module.my_activation_fn".
                "activation_fn": "tensorflow.identity"
            }
        """
        return {
            "activation_fn": "tensorflow.identity"
        }

    def _build(self, inputs): #pylint: disable=W0221
        """Transforms the inputs with an MLP layer and packs the results to have
        the same structure with the decoder state.

        Args:
            inputs: Input (structure of) tensors to be transformed and passed
                to the decoder. Must be a Tensor of shape `[batch_size, ...]`
                or a (nested) tuple of such Tensors.

        Returns:
            A Tensor or a (nested) tuple of Tensors of the same structure of
            the decoder state.
        """
        fn_modules = ['txtgen.custom', 'tensorflow', 'tensorflow.nn']
        activation_fn = get_function(self.hparams.activation_fn, fn_modules)

        output = _mlp_transform(inputs, self._decoder_state_size, activation_fn)

        self._add_internal_trainable_variables()
        self._built = True

        return output


# TODO(zhiting): transform results into the same structure with the decoder
# state
class StochasticConnector(ConnectorBase):
    """Samples decoder initial state from a distribution defined by the inputs.

    Used in, e.g., variational autoencoders, adversarial autoencoders, and other
    models.
    """

    def __init__(self, decoder_state_size, hparams=None,
                 name="stochastic_connector"):
        ConnectorBase.__init__(self, decoder_state_size, hparams, name)

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

    def _build(self, inputs):  # pylint: disable=W0221
        """Samples from a distribution defined by the inputs.

        Args:
            inputs: Result of encoder (e.g., encoder outputs or final
                states) to be transformed and passed to the decoder. Must be a
                Tensor of shape `[batch_size, ...]` or a (nested) tuple of such
                Tensors.

                gaussian_distribution only supports type encoder results. The
                encoder results can either be (mu, logvar) or (mu, logvar,
                context) with size `[batch_size, ...]` for tuple element. When
                context is present, it will be concatenated with the sample
                along the 1st axis.

        Returns:
            A Tensor or a (nested) tuple of Tensors of the same structure of
            the decoder state.
        """
        modules = ['txtgen.custom', 'txtgen.core.distributions']
        sampler = get_function(self.hparams.distribution, modules)

        if sampler is distributions.sample_gaussian:
            if not isinstance(inputs, tuple):
                raise ValueError(
                    "Gaussian connector requires tuple input tensors.")

            if len(inputs) == 2:
                input_mu, input_log_var = inputs
                output = sampler(input_mu, input_log_var)

            elif len(inputs) == 3:
                input_mu, input_log_var, context = inputs
                sample = sampler(input_mu, input_log_var)
                output = tf.concat([sample, context], axis=1)
            else:
                raise ValueError("Gaussian connector supports either "
                                 "(mu, logvar) or (mu, logvar, context)")

        else:
            raise ValueError("Unsupported distribution")

        self._add_internal_trainable_variables()
        self._built = True

        return output

