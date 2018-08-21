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
Various neural networks and related utilities.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

from texar.modules.networks.network_base import FeedForwardNetworkBase
from texar.modules.networks.network_base import _build_layers

__all__ = [
    "FeedForwardNetwork"
]

class FeedForwardNetwork(FeedForwardNetworkBase):
    """Feed-forward neural network that consists of a sequence of layers.

    Args:
        layers (list, optional): A list of :tf_main:`Layer <layers/Layer>`
            instances composing the network. If not given, layers are created
            according to :attr:`hparams`.
        hparams (dict, optional): Embedder hyperparameters. Missing
            hyperparamerter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter sturcture and
            default values.

    See :meth:`~texar.modules.RNNDecoderBase._build` of
    :class:`~texar.modules.FeedForwardNetworkBase` for the inputs and outputs.

    Example:

        .. code-block:: python

            hparams = { # Builds a two-layer dense NN
                "layers": [
                    { "type": "Dense", "kwargs": { "units": 256 },
                    { "type": "Dense", "kwargs": { "units": 10 }
                ]
            }
            nn = FeedForwardNetwork(hparams=hparams)

            inputs = tf.random_uniform([64, 100])
            outputs = nn(inputs)
            # outputs == Tensor of shape [64, 10]
    """

    def __init__(self, layers=None, hparams=None):
        FeedForwardNetworkBase.__init__(self, hparams)

        with tf.variable_scope(self.variable_scope):
            _build_layers(
                self, layers=layers, layer_hparams=self._hparams.layers)

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                "layers": [],
                "name": "NN"
            }

        Here:

        "layers" : list
            A list of layer hyperparameters. See :func:`~texar.core.get_layer`
            for the details of layer hyperparameters.

        "name" : str
            Name of the network.
        """
        return {
            "layers": [],
            "name": "NN"
        }

