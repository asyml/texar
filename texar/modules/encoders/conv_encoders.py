#
"""
Various convolutional network encoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from texar.modules.encoders.encoder_base import EncoderBase
from texar.modules.networks import FeedForwardNetwork
from texar.core import layers

# pylint: disable=not-context-manager, too-many-arguments

__all__ = [
]

def _to_list(value, name=None, list_length=None):
    """Converts hparam value into a list.
    If :attr:`list_length` is given,
    then the canonicalized :attr:`value` must be of
    length :attr:`list_length`.
    """
    if not isinstance(value, (list, tuple)):
        value = [value]
    if list_length is not None and len(value) != list_length:
        raise ValueError("hparams['%s'] must be a list of length %d"
                         % (name, list_length))
    return value

class SimpleConv1DEncoder(EncoderBase):
    """Simple Conv-1D encoder.
    """

    def __init__(self, hparams=None):
        EncoderBase.__init__(self, hparams)



    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.
        """
        return {
            # Conv layers
            "num_conv_layers": 1,
            "filters": 128,
            "kernel_size": [3, 4, 5],
            "conv_activation": "identity",
            "other_conv_kwargs": None,
            # Pooling layers
            "pooling": "MaxPooling1D",
            "pool_size": 1,
            "pool_strides": 1,
            "other_pool_kwargs": None,
            # Dense layers
            "num_dense_layers": 1,
            "dense_size": 128,
            "dense_activation": "identity",
            "other_dense_kwargs": None,
            # Dropout
            "dropout_conv": 1,
            "dropout_dense": [],
            "dropout_rate": 0.75,
            # Others
            "name": "conv_encoder",
            "@no_typecheck": ["filters", "kernel_size",
                              "conv_activation", "dense_activation",
                              "dropout_conv", "dropout_dense"]
        }

    def _build_pooling_hparams(self):
        # "pooling": "MaxPooling1D",
        # "pool_size": 1,
        # "pool_strides": 1,
        # "other_pool_kwargs": None,
        npool = self._hparams.num_conv_layers
        pool_size = _to_list(self._hparams.pool_size, "pool_size", npool)
        strides = _to_list(self._hparams.pool_strides, "pool_strides", npool)

    def _build_conv1d_hparams(self, pooling_hparams):
        """Creates the hparams for each of the conv layers usable for
        :func:`texar.core.layers.get_layer`.
        """
        nconv = self._hparams.num_conv_layers
        filters = _to_list(self._hparams.filters, 'filters', nconv)

        if nconv == 1:
            kernel_size = _to_list(self._hparams.kernel_size)
            if not isinstance(kernel_size[0], (list, tuple)):
                kernel_size = [kernel_size]
        else: # nconv > 1
            kernel_size = _to_list(self._hparams.kernel_size,
                                   'kernel_size', nconv)
            kernel_size = [_to_list(ks) for ks in kernel_size]

        other_kwargs = self._hparams.other_conv_kwargs
        if other_kwargs is not None and not isinstance(other_kwargs, dict):
            raise ValueError("hparams['other_conv_kwargs'] must be a dict.")

        conv_hparams = []
        for i in range(nconv):
            ks_i = kernel_size[i]
            if len(ks_i) > 1: # creates MergeLayer
                for ks_ij in ks_i:
                    conv_kwargs_ij = {
                        "filters": filters[i],
                        "kernel_size": kernel_size[i],
                        "activation": self._hparams.conv_activation
                    }
                    conv_kwargs_ij.update(other_kwargs)
            #conv_hparams.append(
            #    {"type": "Conv1D", "kwargs": conv_kwargs_i})

        return conv_hparams


    def _build(self, inputs):
        pass

