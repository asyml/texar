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

def _mlp_transform(inputs, output_size, activation="identity"):
    """Transforms inputs through a fully-connected layer that creates the output
    with specified size.

    Args:
        inputs: A Tensor of shape `[batch_size, ...]` (i.e., batch-major), or a
            (possibly nested) tuple of such elements. A Tensor or a (possibly
            nested) tuple of Tensors with shape `[max_time, batch_size, ...]`
            (i.e., time-major) can be transposed to batch-major using
            `txtgen.core.utils.transpose_batch_time` prior to this function.
        output_size: Can be an Integer, a TensorShape, or a (possibly nested)
            tuple of Integers or TensorShape.
        activation: Name of activation function applied to the output.

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

    # get activation function
    module_names = ['tensorflow', 'tensorflow.nn']
    activation_fn = get_function(activation, module_names)

    output = tf.contrib.layers.fully_connected(concat_input, sum_output_size)


class ForwardConnector(ConnectorBase):
    """Simply forward the final state of encoder to decoder without
    transformation.

    The encoder and decoder must have the same state structures and sizes.
    """

    def __init__(self, decoder_state_size, name="forward_connector"):
        """Initializes the connector.

        Args:
            decoder_state_size: Size(s) of state(s) of the decoder cell. Can be
                an Integer, a Tensorshape , or a tuple of Integers or
                TensorShapes. Typically can be obtained by
                `decoder.cell.state_size`.
            name: Name of connector.
        """
        ConnectorBase.__init__(decoder_state_size, name, None)

    def _build(self, encoder_state):    #pylint: disable=W0221
        """Passes the encoder outputs to the initial states of decoder.

        Args:
            encoder_state: The state of encoder to pass forward. Must have the
                same sizes with the decoder state.

        Returns:
            The state of the encoder.
        """
        nest.assert_same_structure(encoder_state,
                                   self._decoder_state_size)
        return encoder_state

    @staticmethod
    def default_hparams():
        return {}


class StochasticConnector(ConnectorBase):
    """Samples decoder initial state from a distribution defined by the
    encoder outputs.

    Used in, e.g., variational autoencoders, adversarial autoencoders, and other
    models.
    """
    pass  # TODO
