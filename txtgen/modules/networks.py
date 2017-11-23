#
"""
Various neural networks and related utilities.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

from txtgen import HParams
from txtgen.core.layers import get_layer


class FeedForwardNetwork(object): # pylint: disable=too-many-instance-attributes
    """Feed forward neural network that consists of a sequence of layers.

    Args:
        layers (list, optional):
    """

    def __init__(self, layers=None, hparams=None):
        self._hparams = HParams(hparams, self.default_hparams())
        self._template = tf.make_template(self.hparams.name, self._build,
                                          create_scope_now_=True)
        self._unique_name = self.variable_scope.name.split("/")[-1]

        if layers is not None:
            self._layers = layers
        else:
            self._layers = []
            for layer_id in range(self._hparams.layers):
                self._layers.append(
                    get_layer(self._hparams.layers[layer_id]))

        self._layer_names = []
        self._layers_by_name = {}
        self._layers_outputs_by_name = {}
        self._trainable_variables = []
        self._built = False

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.
        """
        return {
            "layers": [],
            "name": "NN"
        }

    def _build(self, inputs):
        """

        Args:
            inputs:

        Returns:
        """

        #for layer in self._layers:
        #    if layer.scope_name in self._layers_by_name:

        raise NotImplementedError

    def __call__(self, inputs):
        """

        Args:

        Returns:
        """
        raise NotImplementedError

    @property
    def variable_scope(self):
        """The variable scope of the network.
        """
        return self._template.variable_scope

    @property
    def name(self):
        """The uniquified name of the network.
        """
        return self._unique_name

    #@property
    #def trainable_variables(self):
    #    """The list of trainable variables of the module.
    #    """
    #    if not self._built:
    #        raise ValueError(
    #            "Attempting to access trainable_variables before module %s "
    #            "was fully built. The module is built once it is called, "
    #            "e.g., with `%s(...)`" % (self.name, self.name))
    #    return self._trainable_variables

    @property
    def hparams(self):
        """The hyperparameters of the network.

        Returns:
            A :class:`~txtgen.hyperparams.HParams` instance.
        """
        return self._hparams
