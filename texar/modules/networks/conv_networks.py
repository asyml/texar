#
"""
Various convolutional networks.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from texar.modules.networks.network_base import FeedForwardNetworkBase
from texar.modules.networks.network_base import _build_layers
from texar.core.layers import get_pooling_layer_hparams
from texar.utils.utils import uniquify_str
from texar.hyperparams import HParams

# pylint: disable=not-context-manager, too-many-arguments, too-many-locals

__all__ = [
    "_to_list",
    "Conv1DNetwork"
]

def _to_list(value, name=None, list_length=None):
    """Converts hparam value into a list.

    If :attr:`list_length` is given,
    then the canonicalized :attr:`value` must be of
    length :attr:`list_length`.
    """
    if not isinstance(value, (list, tuple)):
        if list_length is not None:
            value = [value] * list_length
        else:
            value = [value]
    if list_length is not None and len(value) != list_length:
        name = '' if name is None else name
        raise ValueError("hparams '%s' must be a list of length %d"
                         % (name, list_length))
    return value

class Conv1DNetwork(FeedForwardNetworkBase):
    """Simple Conv-1D network which consists of a sequence of conv layers
    followed with a sequence of dense layers.
    """

    def __init__(self, hparams=None):
        FeedForwardNetworkBase.__init__(self, hparams)

        with tf.variable_scope(self.variable_scope):
            layer_hparams = self._build_layer_hparams()
            _build_layers(self, layers=None, layer_hparams=layer_hparams)

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        TODO
        """
        return {
            # Conv layers
            "num_conv_layers": 1,
            "filters": 128,
            "kernel_size": [3, 4, 5],
            "conv_activation": "relu",
            "other_conv_kwargs": None,
            # Pooling layers
            "pooling": "MaxPooling1D",
            "pool_size": None,
            "pool_strides": 1,
            "other_pool_kwargs": None,
            # Dense layers
            "num_dense_layers": 1,
            "dense_size": 128,
            "dense_activation": "identity",
            "final_dense_activation": None,
            "other_dense_kwargs": None,
            # Dropout
            "dropout_conv": [1],
            "dropout_dense": [],
            "dropout_rate": 0.75,
            # Others
            "name": "conv1d_network",
            "@no_typecheck": ["filters", "kernel_size", "conv_activation",
                              "pool_size", "pool_strides",
                              "dense_size", "dense_activation",
                              "dropout_conv", "dropout_dense"]
        }

    def _build_pool_hparams(self):
        pool_type = self._hparams.pooling
        if pool_type == "MaxPooling":
            pool_type = "MaxPooling1D"
        elif pool_type == "AveragePooling":
            pool_type = "AveragePooling1D"

        npool = self._hparams.num_conv_layers
        pool_size = _to_list(self._hparams.pool_size, "pool_size", npool)
        strides = _to_list(self._hparams.pool_strides, "pool_strides", npool)

        other_kwargs = self._hparams.other_pool_kwargs or {}
        if isinstance(other_kwargs, HParams):
            other_kwargs = other_kwargs.todict()
        if not isinstance(other_kwargs, dict):
            raise ValueError("hparams['other_pool_kwargs'] must be a dict.")

        pool_hparams = []
        for i in range(npool):
            kwargs_i = {"pool_size": pool_size[i], "strides": strides[i],
                        "name": "pool_%d" % (i+1)}
            kwargs_i.update(other_kwargs)
            pool_hparams_ = get_pooling_layer_hparams({"type": pool_type,
                                                       "kwargs": kwargs_i})
            pool_hparams.append(pool_hparams_)

        return pool_hparams

    def _build_conv1d_hparams(self, pool_hparams):
        """Creates the hparams for each of the conv layers usable for
        :func:`texar.core.layers.get_layer`.
        """
        nconv = self._hparams.num_conv_layers
        if len(pool_hparams) != nconv:
            raise ValueError("`pool_hparams` must be of length %d" % nconv)

        filters = _to_list(self._hparams.filters, 'filters', nconv)
        if nconv == 1:
            kernel_size = _to_list(self._hparams.kernel_size)
            if not isinstance(kernel_size[0], (list, tuple)):
                kernel_size = [kernel_size]
        elif nconv > 1:
            kernel_size = _to_list(self._hparams.kernel_size,
                                   'kernel_size', nconv)
            kernel_size = [_to_list(ks) for ks in kernel_size]

        other_kwargs = self._hparams.other_conv_kwargs or {}
        if isinstance(other_kwargs, HParams):
            other_kwargs = other_kwargs.todict()
        if not isinstance(other_kwargs, dict):
            raise ValueError("hparams['other_conv_kwargs'] must be a dict.")

        conv_pool_hparams = []
        for i in range(nconv):
            hparams_i = []
            names = []
            for ks_ij in kernel_size[i]:
                name = uniquify_str("conv_%d" % (i+1), names)
                names.append(name)
                conv_kwargs_ij = {
                    "filters": filters[i],
                    "kernel_size": ks_ij,
                    "activation": self._hparams.conv_activation,
                    "name": name
                }
                conv_kwargs_ij.update(other_kwargs)
                hparams_i.append(
                    {"type": "Conv1D", "kwargs": conv_kwargs_ij})
            if len(hparams_i) == 1:
                conv_pool_hparams.append([hparams_i[0], pool_hparams[i]])
            else:  # creates MergeLayer
                mrg_kwargs_layers = []
                for hparams_ij in hparams_i:
                    seq_kwargs_j = {"layers": [hparams_ij, pool_hparams[i]]}
                    mrg_kwargs_layers.append(
                        {"type": "SequentialLayer", "kwargs": seq_kwargs_j})
                mrg_hparams = {"type": "MergeLayer",
                               "kwargs": {"layers": mrg_kwargs_layers,
                                          "name": "conv_pool_%d" % (i+1)}}
                conv_pool_hparams.append(mrg_hparams)

        return conv_pool_hparams

    def _build_dense_hparams(self):
        ndense = self._hparams.num_dense_layers
        dense_size = _to_list(self._hparams.dense_size, 'dense_size', ndense)

        other_kwargs = self._hparams.other_dense_kwargs or {}
        if isinstance(other_kwargs, HParams):
            other_kwargs = other_kwargs.todict()
        if not isinstance(other_kwargs, dict):
            raise ValueError("hparams['other_dense_kwargs'] must be a dict.")

        dense_hparams = []
        for i in range(ndense):
            activation = self._hparams.dense_activation
            if i == ndense - 1 and not self._hparams.final_dense_activation:
                activation = self._hparams.final_dense_activation

            kwargs_i = {"units": dense_size[i],
                        "activation": activation,
                        "name": "dense_%d" % (i+1)}
            kwargs_i.update(other_kwargs)

            dense_hparams.append({"type": "Dense", "kwargs": kwargs_i})

        return dense_hparams

    def _build_layer_hparams(self):
        pool_hparams = self._build_pool_hparams()
        conv_pool_hparams = self._build_conv1d_hparams(pool_hparams)
        dense_hparams = self._build_dense_hparams()

        def _dropout_hparams(layer_id):
            return {"type": "Dropout",
                    "kwargs": {"rate": self._hparams.dropout_rate,
                               "name": "dropout_%d" % layer_id}}
        dropout_conv = _to_list(self._hparams.dropout_conv)
        dropout_dense = _to_list(self._hparams.dropout_dense)

        layers_hparams = []
        nconv = self._hparams.num_conv_layers
        for conv_i in range(nconv):
            if conv_i in dropout_conv:
                layers_hparams.append(_dropout_hparams(conv_i))
            if isinstance(conv_pool_hparams[conv_i], (list, tuple)):
                layers_hparams += conv_pool_hparams[conv_i]
            else:
                layers_hparams.append(conv_pool_hparams[conv_i])
        if nconv in dropout_conv:
            layers_hparams.append(_dropout_hparams(nconv))

        ndense = self._hparams.num_dense_layers
        if ndense > 0: # Add flatten layers before dense layers
            layers_hparams.append({"type": "Flatten"})
        for dense_i in range(ndense):
            if dense_i in dropout_dense:
                layers_hparams.append(_dropout_hparams(dense_i + nconv))
            layers_hparams.append(dense_hparams[dense_i])
        if ndense in dropout_dense:
            layers_hparams.append(_dropout_hparams(ndense + nconv))

        return layers_hparams


