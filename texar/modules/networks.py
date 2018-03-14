#
"""
Various neural networks and related utilities.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

from texar import context
from texar.modules.module_base import ModuleBase
from texar.core.layers import get_layer
from texar.core.utils import uniquify_str

# pylint: disable=too-many-instance-attributes, arguments-differ

__all__ = [
    "FeedForwardNetwork"
]

#TDDO(zhiting): complete the docs
class FeedForwardNetwork(ModuleBase):
    """Feed forward neural network that consists of a sequence of layers.

    Args:
        layers (list, optional):
    """

    def __init__(self, layers=None, hparams=None):
        ModuleBase.__init__(self, hparams)

        self._layers = []
        self._layer_names = []
        self._layers_by_name = {}
        self._layer_outputs = []
        self._layer_outputs_by_name = {}

        self._build_layers(layers)

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        TODO
        """
        return {
            "layers": [],
            "name": "NN"
        }

    def _build_layers(self, layers):
        """Builds layers.
        """
        with tf.variable_scope(self.variable_scope):
            if layers is not None:
                self._layers = layers
            else:
                self._layers = []
                for layer_id in range(len(self._hparams.layers)):
                    self._layers.append(
                        get_layer(hparams=self._hparams.layers[layer_id]))

        for layer in self._layers:
            layer_name = uniquify_str(layer.name, self._layer_names)
            self._layer_names.append(layer_name)
            self.layers_by_name[layer_name] = layer

    def _build(self, inputs, mode=None):
        """

        Args:
            inputs:

        Returns:
        """
        training = context.is_train()
        if mode is not None and mode == tf.estimator.ModeKeys.TRAIN:
            training = True

        prev_outputs = inputs
        for layer_id, layer in enumerate(self._layers):
            if isinstance(layer, tf.layers.Dropout) or \
                    isinstance(layer, tf.layers.BatchNormalization):
                outputs = layer(prev_outputs, training=training)
            else:
                outputs = layer(prev_outputs)
            self._layer_outputs.append(outputs)
            self._layer_outputs_by_name[self._layer_names[layer_id]] = outputs
            prev_outputs = outputs

        if not self._built:
            self._add_internal_trainable_variables()
            # Add trainable variables of `self._layers` which may be constructed
            # externally.
            for layer in self._layers:
                self._add_trainable_variable(layer.trainable_variables)
            self._built = True

        return outputs


    def has_layer(self, layer_name):
        """Returns `True` if the network with the name exists. Returns `False`
        otherwise.

        Args:
            layer_name (str): Name of the layer.
        """
        return layer_name in self._layers_by_name

    def layer_by_name(self, layer_name):
        """Returns the layer with the name. Returns 'None' if the layer name
        does not exist.

        Args:
            layer_name (str): Name of the layer.
        """
        return self._layers_by_name.get(layer_name, None)

    @property
    def layers_by_name(self):
        """A dictionary mapping layer names to the layers.
        """
        return self._layers_by_name

    @property
    def layers(self):
        """A list of the layers.
        """
        return self._layers

    @property
    def layer_names(self):
        """A list of uniquified layer names.
        """
        return self._layer_names

    def layer_outputs_by_name(self, layer_name):
        """Returns the output tensors of the layer with the specified name.
        Returns `None` if the layer name does not exist.

        Args:
            layer_name (str): Name of the layer.
        """
        return self._layer_outputs_by_name.get(layer_name, None)

    @property
    def layer_outputs(self):
        """A list containing output tensors of each layer.
        """
        return self._layer_outputs

