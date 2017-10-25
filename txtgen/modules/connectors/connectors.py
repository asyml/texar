#
"""
Various connectors.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.distributions as tf_dstr
from tensorflow.python.util import nest    # pylint: disable=E0611

from txtgen.modules.connectors.connector_base import ConnectorBase
from txtgen.core.utils import get_function
from txtgen.core.utils import get_instance

# pylint: disable=too-many-locals, arguments-differ, too-many-arguments

#TODO(zhiting): updates docs to not restrict to "output decoder state", but
# instead output Tensors of any specified size

__all__ = [
    "ConstantConnector", "ForwardConnector", "MLPTransformConnector",
    "ReparameterizedStochasticConnector", "StochasticConnector",
    "ConcatConnector"
]

def _assert_same_size(outputs, output_size):
    """Check if outputs match output_size

    Args:
        outputs: A Tensor or a (nested) tuple of tensors
        output_size: Can be an Integer, a TensorShape, or a (nested) tuple of
            Integers or TensorShape.
    """
    nest.assert_same_structure(outputs, output_size)
    flat_output_size = nest.flatten(output_size)
    flat_output = nest.flatten(outputs)

    for (output, size) in zip(flat_output, flat_output_size):
        if output[0].shape != tf.TensorShape(size):
            raise ValueError(
                "The output size does not match the the required output_size")


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
    if isinstance(flat_output_size[0], tf.TensorShape):
        size_list = [0] * len(flat_output_size)
        for (i, shape) in enumerate(flat_output_size):
            size_list[i] = reduce(lambda x, y: x*y,
                                  [dim.value for dim in shape])
    else:
        size_list = flat_output_size
    sum_output_size = sum(size_list)

    fc_output = tf.contrib.layers.fully_connected(
        concat_input, sum_output_size, activation_fn=activation_fn)

    flat_output = tf.split(fc_output, size_list, axis=1)

    if isinstance(flat_output_size[0], tf.TensorShape):
        for (i, shape) in enumerate(flat_output_size):
            new_shape = tf.TensorShape(batch_size).concatenate(shape)
            flat_output[i] = tf.reshape(flat_output[i], new_shape)
    output = nest.pack_sequence_as(structure=output_size,
                                   flat_sequence=flat_output)

    return output


class ConstantConnector(ConnectorBase):
    """Creates decoder initial state that has a constant value.

    Args:
        output_size: Size of state of the decoder cell. Can be an
            Integer, a Tensorshape, or a tuple of Integers or TensorShapes.
            This can typically be obtained by :attr:`decoder.state_size`.
        hparams (dict): Hyperparameters of the connector.
    """
    def __init__(self, output_size, hparams=None):
        ConnectorBase.__init__(self, output_size, hparams)

    @staticmethod
    def default_hparams():
        """Returns a dictionary of default hyperparameters.

        Returns:
            .. code-block:: python

                {
                    "value": 0.,
                    "name": "constant_connector"
                }

            Here:

            "value": float
                The constant value that the output tensor has.

                The default value is `0.`.

            "name": str
                Name of the connector.

                The default value is "constant_connector".
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
            self._output_size)
        return output


class ForwardConnector(ConnectorBase):
    """Directly forwards input (structure of) tensors to decoder.

    The input must have the same structure with the decoder state,
    or must have the same number of elements and be re-packable into the decoder
    state structure. Note that if input is or contains a `dict` instance, the
    keys will be sorted to pack in deterministic order (See
    :meth:`~tensorflow.python.util.nest.pack_sequence_as` for more details).

    Args:
        output_size: Size of state of the decoder cell. Can be an
            Integer, a Tensorshape , or a tuple of Integers or TensorShapes.
            This can typically be obtained by :attr:`decoder.cell.state_size`.
    """

    def __init__(self, output_size):
        ConnectorBase.__init__(self, output_size, None)

    @staticmethod
    def default_hparams():
        """Returns a dictionary of default hyperparameters.

        Returns:
            .. code-block:: python

                {
                    "name": "forward_connector"
                }

            Here:

            "name" : str
                Name of the connector.

                The default value is "forward_connector".
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
            nest.assert_same_structure(inputs, self._output_size)
        except (ValueError, TypeError):
            flat_input = nest.flatten(inputs)
            output = nest.pack_sequence_as(
                self._output_size, flat_input)

        self._built = True

        return output


class MLPTransformConnector(ConnectorBase):
    """Transforms inputs with an MLP layer. Takes the outputs as the decoder
    initial state.

    Args:
        output_size: Size of state of the decoder cell. Can be an
            Integer, a Tensorshape , or a tuple of Integers or TensorShapes.
            This can typically be obtained by :attr:`decoder.cell.state_size`.
        hparams (dict): Hyperparameters of the connector.
    """

    def __init__(self, output_size, hparams=None):
        ConnectorBase.__init__(self, output_size, hparams)

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        Returns:
            .. code-block:: python

                {
                    "activation_fn": "tensorflow.identity",
                    "name": "mlp_connector"
                }

            Here:

            "activation_fn" : str
                The name or full path to the activation function applied to
                the outputs of the MLP layer. The activation functions can be:

                - Built-in activation functions defined in `tensorflow` or \
                `tensorflow.nn`, e.g., :meth:`tensorflow.identity`.
                - User-defined activation functions in `txtgen.custom`.
                - External activation functions. Must provide the full path, \
                  e.g., "my_module.my_activation_fn".

                The default value is "tensorflow.identity", i.e., the MLP
                transformation is linear.

            "name" : str
                Name of the connector.

                The default value is "mlp_connector".
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

        output = _mlp_transform(inputs, self._output_size, activation_fn)

        self._add_internal_trainable_variables()
        self._built = True

        return output


class ReparameterizedStochasticConnector(ConnectorBase):
    """Samples from a distribution with reparameterization trick and transforms
    samples into specified size.

    Reparameterization allows gradients to be propagated through the
    stochastic samples. Used in, e.g., Variational Autoencoders (VAEs).

    Args:
        output_size: Size of output. Can be an int, a tuple of int, a
            Tensorshape, or a tuple of TensorShapes. For example, to transform
            to decoder state size, set `output_size=decoder.cell.state_size`.
        hparams (dict): Hyperparameters of the connector.
    """

    def __init__(self, output_size, hparams=None):
        ConnectorBase.__init__(self, output_size, hparams)

    #TODO(zhiting): add docs
    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        Returns:
            .. code-block:: python

                {
                }

            Here:


        """
        return {
            "distribution": {
                "type": "MultivariateNormalDiag",
                "kwargs": {}
            },
            "activation_fn": "tensorflow.identity",
            "name": "reparameterized_stochastic_connector"
        }

    def _build(self,
               distribution=None,
               distribution_type=None,
               distribution_kwargs=None,
               transform=True,
               num_samples=None):
        """Samples from a distribution and optionally performs transformation.

        The distribution must be reparameterizable, i.e.,
        `distribution.reparameterization_type = FULLY_REPARAMETERIZED`.

        Args:
            distribution (optional): An instance of
                :class:`~tensorflow.contrib.distributions.Distribution`. If
                `None` (default), distribution is constructed based on
                :attr:`distribution_type` or
                :attr:`hparams['distribution']['type']`.
            distribution_type (str, optional): Name or path to the distribution
                class which inherits
                :class:`~tensorflow.contrib.distributions.Distribution`. Ignored
                if :attr:`distribution` is specified.
            distribution_kwargs (dict, optional): Keyword arguments of the
                distribution class specified in :attr:`distribution_type`.
            transform (bool): Whether to perform MLP transformation of the
                samples. If `False`, the shape of a sample must match the
                :attr:`output_size`.
            num_samples (int or scalar int Tensor, optional): Number of samples to
                generate. `None` is required in training stage.

        Returns:
            If `num_samples`==None, returns a Tensor of shape `[batch_size x
            output_size]`, else returns a Tensor of shape `[num_samples x
            output_size]`. `num_samples` should be specified if not in
            training stage.

        Raises:
            ValueError: If distribution cannot be reparametrized.
            ValueError: The output does not match the :attr:`output_size`.
        """
        if distribution:
            dstr = distribution
        elif distribution_type and distribution_kwargs:
            dstr = get_instance(
                distribution_type, distribution_kwargs,
                ["txtgen.custom", "tensorflow.contrib.distributions"])
        else:
            dstr = get_instance(
                self.hparams.distribution.type,
                self.hparams.distribution.kwargs,
                ["txtgen.custom", "tensorflow.contrib.distributions"])

        if dstr.reparameterization_type == tf_dstr.NOT_REPARAMETERIZED:
            raise ValueError(
                "Distribution is not reparameterized: %s" % dstr.name)

        if num_samples:
            output = dstr.sample(num_samples)
        else:
            output = dstr.sample()

        if dstr.event_shape == []:
            output = tf.reshape(output,
                                output.shape.concatenate(tf.TensorShape(1)))

        output = tf.cast(output, tf.float32)
        if transform:
            fn_modules = ['txtgen.custom', 'tensorflow', 'tensorflow.nn']
            activation_fn = get_function(self.hparams.activation_fn, fn_modules)
            output = _mlp_transform(output, self._output_size, activation_fn)
        _assert_same_size(output, self._output_size)

        self._add_internal_trainable_variables()
        self._built = True

        return output

class StochasticConnector(ConnectorBase):
    """Samples decoder initial state from a distribution
    defined by the inputs. (disable reparameterize trick)

    Used in, e.g., variational autoencoders, adversarial autoencoders, and other
    models.
    """

    def __init__(self, output_size, hparams=None):
        ConnectorBase.__init__(self, output_size, hparams)

    #TODO(zhiting): add docs
    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        Returns:
            .. code-block:: python

                {
                }

            Here:


        """
        return {
            "distribution": {
                "type": "tf.contrib.distributions.Categorical",
                "kwargs": {}
                },
            "activation_fn": "tensorflow.identity",
            "name": "stochastic_connector"
        }

    # pylint: disable=arguments-differ

    def _build(self,
               distribution=None,
               distribution_type=None,
               distribution_kwargs=None,
               transform=False,
               num_samples=None):

        """Samples from a distribution and optionally performs transformation.

        Gradients would not propagate through the random samples.
        Args:
            distribution (optional): An instance of
                :class:`~tensorflow.contrib.distributions.Distribution`. If
                `None` (default), distribution is constructed based on
                :attr:`distribution_type` or
                :attr:`hparams['distribution']['type']`.
            distribution_type (str, optional): Name or path to the distribution
                class which inherits
                :class:`~tensorflow.contrib.distributions.Distribution`. Ignored
                if :attr:`distribution` is specified.
            distribution_kwargs (dict, optional): Keyword arguments of the
                distribution class specified in :attr:`distribution_type`.
            transform (bool): Whether to perform MLP transformation of the
                samples. If `False`, the shape of a sample must match the
                :attr:`output_size`.
            num_samples (int or scalar int Tensor, optional): Number of samples
                to generate. `None` is required in training stage.

        Returns:
            If `num_samples`==None, returns a Tensor of shape `[batch_size x
            output_size]`, else returns a Tensor of shape `[num_samples x
            output_size]`. `num_samples` should be specified if not in
            training stage.

        Raises:
            ValueError: The output does not match the :attr:`output_size`.
        """
        if distribution:
            dstr = distribution
        elif distribution_type and distribution_kwargs:
            dstr = get_instance(
                distribution_type, distribution_kwargs,
                ["txtgen.custom", "tensorflow.contrib.distributions"])
        else:
            dstr = get_instance(
                self.hparams.distribution.type,
                self.hparams.distribution.kwargs,
                ["txtgen.custom", "tensorflow.contrib.distributions"])

        if num_samples:
            output = dstr.sample(num_samples)
        else:
            output = dstr.sample()

        if dstr.event_shape == []:
            output = tf.reshape(output,
                                output.shape.concatenate(tf.TensorShape(1)))

        # Disable gradients through samples
        output = tf.stop_gradient(output)

        output = tf.cast(output, tf.float32)

        if transform:
            fn_modules = ['txtgen.custom', 'tensorflow', 'tensorflow.nn']
            activation_fn = get_function(self.hparams.activation_fn, fn_modules)
            output = _mlp_transform(output, self._output_size, activation_fn)
        _assert_same_size(output, self._output_size)

        self._add_internal_trainable_variables()
        self._built = True

        return output

class ConcatConnector(ConnectorBase):
    """ Concatenate multiple connectors into one connector.
    Used in, e.g., semi-supervised variational autoencoders,
    disentangled representation learning, and other
    models.
    """

    def __init__(self, output_size, hparams=None):
        ConnectorBase.__init__(self, output_size, hparams)

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        Returns:
            .. code-block:: python

                {
                }

            Here:


        """
        return {
            "activation_fn": "tensorflow.identity",
            "name": "concat_connector"
        }

    def _build(self, connector_inputs, transform=True):  # pylint: disable=W0221
        """Concatenate multiple input connectors

        Args:
            connector_inputs: a list of connector states

        Returns:
            A Tensor or a (nested) tuple of Tensors of the same structure of
            the decoder state.
        """

        connector_inputs = [tf.cast(connector, tf.float32)
                            for connector in connector_inputs]
        output = tf.concat(connector_inputs, axis=1)

        if transform:
            fn_modules = ['txtgen.custom', 'tensorflow', 'tensorflow.nn']
            activation_fn = get_function(self.hparams.activation_fn, fn_modules)
            output = _mlp_transform(output, self._output_size, activation_fn)
        _assert_same_size(output, self._output_size)

        self._add_internal_trainable_variables()
        self._built = True

        return output
