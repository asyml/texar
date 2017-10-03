#
"""
Various connectors.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.distributions as tfds
from tensorflow.python.util import nest    # pylint: disable=E0611

from txtgen.modules.connectors.connector_base import ConnectorBase
from txtgen.core.utils import get_function


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
            This can typically be obtained by :attr:`decoder.state_size`.
        hparams (dict): Hyperparameters of the connector.
    """
    def __init__(self, decoder_state_size, hparams=None):
        ConnectorBase.__init__(self, decoder_state_size, hparams)

    @staticmethod
    def default_hparams():
        """Returns a dictionary of default hyperparameters:

        .. code-block:: python

            {
                # The constant value that the decoder initial state has.
                "value": 0.,
                # The name of the connector.
                "name": "constant_connector"
            }
        """
        return {
            "value": 0.,
            "name": "constant_connector"
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
    """

    def __init__(self, decoder_state_size):
        ConnectorBase.__init__(self, decoder_state_size, None)

    @staticmethod
    def default_hparams():
        """Returns a dictionary of default hyperparameters.

        .. code-block:: python

            {
                # The name of the connector.
                "name": "forward_connector"
        """
        return {
            "name": "forward_connector"
        }

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

    def __init__(self, decoder_state_size, hparams=None):
        ConnectorBase.__init__(self, decoder_state_size, hparams)

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
                "activation_fn": "tensorflow.identity",

                # Name of the connector.
                "name": "mlp_connector"
            }
        """
        return {
            "activation_fn": "tensorflow.identity",
            "name": "mlp_connector"
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


class ReparameterizedStochasticConnector(ConnectorBase):
    """Samples decoder initial state using reparameterization trick
    from a distribution defined by the inputs.

    Used in, e.g., variational autoencoders, adversarial autoencoders, and other
    models.
    """

    def __init__(self, decoder_state_size, hparams=None):
        ConnectorBase.__init__(self, decoder_state_size, hparams)

    #TODO(zhiting): add docs
    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        Returns:
            ```python
            {
            }
            ```
        """
        return {
            "distribution": "tf.contrib.distributions.MultivariateNormalDiag",
            "name": "reparameterized_stochastic_connector"
        }

    def _build(self, distribution, batch_size):  # pylint: disable=W0221
        """Samples from a distribution defined by the inputs.

        Args:
            distribution: Instance of tf.contrib.distributions
            batch_size (int or scalar int Tensor): The batch size.

        Returns:
            A Tensor or a (nested) tuple of Tensors of the same structure of
            the decoder state.
inputs
        Raises:
            ValueError: An error occurred when the input distribution cannot be reparameterized
        """

        if distribution.reparameterization_type == tfds.NOT_REPARAMETERIZED:
            raise ValueError("%s distribution is not reparameterized" % distribution.name)

        output = distribution.sample(batch_size)
        if len(output.get_shape()) == 1:
            output = output.reshape([batch_size, 1])

        # TODO(junxian): transform to decoder state size
        # try:
            # nest.assert_same_structure(output, self._decoder_state_size)
        # except (ValueError, TypeError):
            # flat_output = nest.flatten(output)
            # output = nest.pack_sequence_as(
                # self._decoder_state_size, flat_output)

        self._add_internal_trainable_variables()
        self._built = True

        return output

class StochasticConnector(ConnectorBase):
    """Samples decoder initial state from a distribution
    defined by the inputs. (disable reparameterize trick)

    Used in, e.g., variational autoencoders, adversarial autoencoders, and other
    models.
    """

    def __init__(self, decoder_state_size, hparams=None):
        ConnectorBase.__init__(self, decoder_state_size, hparams)

    #TODO(zhiting): add docs
    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        Returns:
            ```python
            {
            }
            ```
        """
        return {
            "distribution": "tf.contrib.distributions.Categorical",
            "name": "stochastic_connector"
        }

    def _build(self, distribution, batch_size):  # pylint: disable=W0221
        """Samples from a distribution defined by the inputs.

        Args:
            distribution: Instance of tf.contrib.distributions
            batch_size (int or scalar int Tensor): The batch size.

        Returns:
            A Tensor or a (nested) tuple of Tensors of the same structure of
            the decoder state.
        """

        output = distribution.sample(batch_size)
        if len(output.get_shape()) == 1:
            output = tf.reshape(output, [batch_size, 1])

        # Disable gradients through samples
        output = tf.stop_gradient(output)

        # TODO(junxian): transform to decoder size
        # try:
            # nest.assert_same_structure(output, self._decoder_state_size)
        # except (ValueError, TypeError):
            # flat_output = nest.flatten(output)
            # output = nest.pack_sequence_as(
                # self._decoder_state_size, flat_output)

        self._add_internal_trainable_variables()
        self._built = True

        return output

class ConcatConnector(ConnectorBase):
    """ Concatenate multiple connectors into one connector.
    Used in, e.g., semi-supervised variational autoencoders,
    disentangled representation learning, and other
    models.
    """

    def __init__(self, decoder_state_size, hparams=None):
        ConnectorBase.__init__(self, decoder_state_size, hparams)

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        Returns:
            ```python
            {
            }
            ```
        """
        return {
            "name": "concatconnector"
        }

    def _build(self, connector_inputs):  # pylint: disable=W0221
        """ Concatenate multiple input connectors

        Args:
            connector_inputs: a list of connector states

        Returns:
            A Tensor or a (nested) tuple of Tensors of the same structure of
            the decoder state.
        """

        connector_inputs = [tf.cast(connector, tf.float32) for connector in connector_inputs]
        output = tf.concat(connector_inputs, axis=1)

        # TODO(junxian): transform to decoder state size
        # try:
            # nest.assert_same_structure(concat_output, self._decoder_state_size)
        # except (ValueError, TypeError):
            # flat_output = nest.flatten(concat_output)
            # output = nest.pack_sequence_as(
                # self._decoder_state_size, flat_output)

        self._add_internal_trainable_variables()
        self._built = True

        return output
