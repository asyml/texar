# Copyright 2019 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Base class for feed forward neural networks.
"""

import tensorflow as tf

from texar.tf.module_base import ModuleBase
from texar.tf.core.layers import get_layer
from texar.tf.utils.mode import is_train_mode
from texar.tf.utils.utils import uniquify_str

# pylint: disable=protected-access

__all__ = [
    "_build_layers",
    "FeedForwardNetworkBase"
]


def _build_layers(network, layers=None, layer_hparams=None):
    r"""Builds layers.

    Either :attr:`layer_hparams` or :attr:`layers` must be
    provided. If both are given, :attr:`layers` will be used.

    Args:
        network: An instance of a subclass of
            :class:`~texar.tf.modules.networks.FeedForwardNetworkBase`
        layers (optional): A list of layer instances.
        layer_hparams (optional): A list of layer hparams, each to which
            is fed to :func:`~texar.tf.core.layers.get_layer` to create the
            layer instance.
    """
    if layers is not None:
        network._layers = layers
    else:
        if layer_hparams is None:
            raise ValueError(
                'Either `layer` or `layer_hparams` is required.')
        network._layers = []
        for _, hparams in enumerate(layer_hparams):
            network._layers.append(get_layer(hparams=hparams))

    for layer in network._layers:
        layer_name = uniquify_str(layer.name, network._layer_names)
        network._layer_names.append(layer_name)
        network._layers_by_name[layer_name] = layer


class FeedForwardNetworkBase(ModuleBase):
    r"""Base class inherited by all feed-forward network classes.

    Args:
        hparams (dict, optional): Hyperparameters. Missing
            hyperparamerter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter sturcture and
            default values.

    See :meth:`call` for the inputs and outputs.
    """

    def __init__(self, hparams=None):
        super().__init__(hparams=hparams)

        self._layers = []
        self._layer_names = []
        self._layers_by_name = {}
        self._layer_outputs = []
        self._layer_outputs_by_name = {}

    @staticmethod
    def default_hparams():
        r"""Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                "name": "NN"
            }
        """
        return {
            "name": "NN"
        }

    def call(self, inputs, mode=None):
        r"""Feeds forward inputs through the network layers and returns outputs.

        Args:
            inputs: The inputs to the network. The requirements on inputs
                depends on the first layer and subsequent layers in the
                network.
            mode (optional): A tensor taking value in
                :tf_main:`tf.estimator.ModeKeys <estimator/ModeKeys>`, including
                `TRAIN`, `EVAL`, and `PREDICT`.

        Returns:
            The output of the network.
        """
        training = is_train_mode(mode)

        prev_outputs = inputs
        for layer_id, layer in enumerate(self._layers):
            if isinstance(layer, (tf.keras.layers.Dropout,
                                  tf.keras.layers.BatchNormalization)):
                outputs = layer(prev_outputs, training=training)
            else:
                outputs = layer(prev_outputs)
            self._layer_outputs.append(outputs)
            self._layer_outputs_by_name[self._layer_names[layer_id]] = outputs
            prev_outputs = outputs

        return outputs

    def append_layer(self, layer):
        r"""Appends a layer to the end of the network. The method is only
        feasible before :attr:`call` is called.

        Args:
            layer: A :tf_main:`tf.keras.layers.Layer <layers/Layer>` instance,
                or a dict of layer hyperparameters.
        """
        if self.built:
            raise ValueError("`FeedForwardNetwork.append_layer` can be "
                             "called only before `call` is called.")

        layer_ = layer
        if not isinstance(layer_, tf.keras.layers.Layer):
            layer_ = get_layer(hparams=layer_)
        self._layers.append(layer_)
        layer_name = uniquify_str(layer_.name, self._layer_names)
        self._layer_names.append(layer_name)
        self._layers_by_name[layer_name] = layer_

    def has_layer(self, layer_name):
        r"""Returns `True` if the network with the name exists. Returns `False`
        otherwise.

        Args:
            layer_name (str): Name of the layer.
        """
        return layer_name in self._layers_by_name

    def layer_by_name(self, layer_name):
        r"""Returns the layer with the name. Returns 'None' if the layer name
        does not exist.

        Args:
            layer_name (str): Name of the layer.
        """
        return self._layers_by_name.get(layer_name, None)

    @property
    def layers_by_name(self):
        r"""A dictionary mapping layer names to the layers.
        """
        return self._layers_by_name

    @property
    def layers(self):
        r"""A list of the layers.
        """
        return self._layers

    @property
    def layer_names(self):
        r"""A list of uniquified layer names.
        """
        return self._layer_names

    def layer_outputs_by_name(self, layer_name):
        r"""Returns the output tensors of the layer with the specified name.
        Returns `None` if the layer name does not exist.

        Args:
            layer_name (str): Name of the layer.
        """
        return self._layer_outputs_by_name.get(layer_name, None)

    @property
    def layer_outputs(self):
        r"""A list containing output tensors of each layer.
        """
        return self._layer_outputs
