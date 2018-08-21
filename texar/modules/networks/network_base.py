# Copyright 2018 The Texar Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

from texar.module_base import ModuleBase
from texar.utils import TexarError
from texar.core.layers import get_layer
from texar.utils.utils import uniquify_str
from texar.utils.mode import is_train_mode

# pylint: disable=too-many-instance-attributes, arguments-differ
# pylint: disable=protected-access

__all__ = [
    "_build_layers",
    "FeedForwardNetworkBase"
]

def _build_layers(network, layers=None, layer_hparams=None):
    """Builds layers.

    Either :attr:`layer_hparams` or :attr:`layers` must be
    provided. If both are given, :attr:`layers` will be used.

    Args:
        network: An instance of a subclass of
            :class:`~texar.modules.networks.network_base.FeedForwardNetworkBase`
        layers (optional): A list of layer instances.
        layer_hparams (optional): A list of layer hparams, each to which
            is fed to :func:`~texar.core.layers.get_layer` to create the
            layer instance.
    """
    with tf.variable_scope(network.variable_scope):
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
    """Base class inherited by all feed-forward network classes.

    Args:
        hparams (dict, optional): Hyperparameters. Missing
            hyperparamerter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter sturcture and
            default values.

    See :meth:`_build` for the inputs and outputs.
    """

    def __init__(self, hparams=None):
        ModuleBase.__init__(self, hparams)

        self._layers = []
        self._layer_names = []
        self._layers_by_name = {}
        self._layer_outputs = []
        self._layer_outputs_by_name = {}

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                "name": "NN"
            }
        """
        return {
            "name": "NN"
        }

    def _build(self, inputs, mode=None):
        """Feeds forward inputs through the network layers and returns outputs.

        Args:
            inputs: The inputs to the network. The requirements on inputs
                depends on the first layer and subsequent layers in the
                network.
            mode (optional): A tensor taking value in
                :tf_main:`tf.estimator.ModeKeys <estimator/ModeKeys>`, including
                `TRAIN`, `EVAL`, and `PREDICT`. If `None`,
                :func:`texar.global_mode` is used.

        Returns:
            The output of the network.
        """
        training = is_train_mode(mode)

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
            # Add trainable variables of `self._layers` which may be
            # constructed externally.
            for layer in self._layers:
                self._add_trainable_variable(layer.trainable_variables)
            self._built = True

        return outputs

    def append_layer(self, layer):
        """Appends a layer to the end of the network. The method is only
        feasible before :attr:`_build` is called.

        Args:
            layer: A :tf_main:`tf.layers.Layer <layers/Layer>` instance, or
                a dict of layer hyperparameters.
        """
        if self._built:
            raise TexarError("`FeedForwardNetwork.append_layer` can be "
                             "called only before `_build` is called.")

        with tf.variable_scope(self.variable_scope):
            layer_ = layer
            if not isinstance(layer_, tf.layers.Layer):
                layer_ = get_layer(hparams=layer_)
            self._layers.append(layer_)
            layer_name = uniquify_str(layer_.name, self._layer_names)
            self._layer_names.append(layer_name)
            self._layers_by_name[layer_name] = layer_

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
