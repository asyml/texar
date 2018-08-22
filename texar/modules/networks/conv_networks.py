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
Various convolutional networks.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from texar.modules.networks.network_base import FeedForwardNetworkBase
from texar.modules.networks.network_base import _build_layers
from texar.core.layers import get_pooling_layer_hparams, get_activation_fn
from texar.utils.utils import uniquify_str
from texar.utils.shapes import mask_sequences
from texar.hyperparams import HParams

# pylint: disable=too-many-arguments, too-many-locals

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

    Args:
        hparams (dict, optional): Hyperparameters. Missing
            hyperparamerter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter sturcture and
            default values.

    See :meth:`_build` for the inputs and outputs. The inputs must be a
    3D Tensor of shape `[batch_size, length, channels]` (default), or
    `[batch_size, channels, length]` (if `data_format` is set to
    `'channels_last'` through :attr:`hparams`). For example, for sequence
    classification, `length` corresponds to time steps, and `channels`
    corresponds to embedding dim.

    Example:

        .. code-block:: python

            nn = Conv1DNetwork() # Use the default structure

            inputs = tf.random_uniform([64, 20, 256])
            outputs = nn(inputs)
            # outputs == Tensor of shape [64, 128], cuz the final dense layer
            # has size 128.

    .. document private functions
    .. automethod:: _build
    """

    def __init__(self, hparams=None):
        FeedForwardNetworkBase.__init__(self, hparams)

        with tf.variable_scope(self.variable_scope):
            layer_hparams = self._build_layer_hparams()
            _build_layers(self, layers=None, layer_hparams=layer_hparams)

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                # (1) Conv layers
                "num_conv_layers": 1,
                "filters": 128,
                "kernel_size": [3, 4, 5],
                "conv_activation": "relu",
                "conv_activation_kwargs": None,
                "other_conv_kwargs": None,
                # (2) Pooling layers
                "pooling": "MaxPooling1D",
                "pool_size": None,
                "pool_strides": 1,
                "other_pool_kwargs": None,
                # (3) Dense layers
                "num_dense_layers": 1,
                "dense_size": 128,
                "dense_activation": "identity",
                "dense_activation_kwargs": None,
                "final_dense_activation": None,
                "final_dense_activation_kwargs": None,
                "other_dense_kwargs": None,
                # (4) Dropout
                "dropout_conv": [1],
                "dropout_dense": [],
                "dropout_rate": 0.75,
                # (5) Others
                "name": "conv1d_network",
            }

        Here:

        1. For **convolutional** layers:

            "num_conv_layers" : int
                Number of convolutional layers.

            "filters" : int or list
                The number of filters in the convolution, i.e., the
                dimensionality
                of the output space. If "num_conv_layers" > 1, "filters" must be
                a list of "num_conv_layers" integers.

            "kernel_size" : int or list
                Lengths of 1D convolution windows.

                - If "num_conv_layers" == 1, this can be a list of arbitrary \
                number\
                of `int` denoting different sized conv windows. The number of \
                filters of each size is specified by "filters". For example,\
                the default values will create 3 sets of filters, each of which\
                has kernel size of 3, 4, and 5, respectively, and has filter\
                number 128.
                - If "num_conv_layers" > 1, this must be a list of length \
                "num_conv_layers". Each element can be an `int` or a list \
                of arbitrary number of `int` denoting the kernel size of \
                respective layer.

            "conv_activation": str or callable
                Activation function applied to the output of the convolutional
                layers. Set to "indentity" to maintain a linear activation.
                See :func:`~texar.core.get_activation_fn` for more details.

            "conv_activation_kwargs" : dict, optional
                Keyword arguments for conv layer activation functions.
                See :func:`~texar.core.get_activation_fn` for more details.

            "other_conv_kwargs" : dict, optional
                Other keyword arguments for
                :tf_main:`tf.layers.Conv1D <layers/Conv1d>` constructor, e.g.,
                "data_format", "padding", etc.

        2. For **pooling** layers:

            "pooling" : str or class or instance
                Pooling layer after each of the convolutional layer(s). Can
                a pooling layer class, its name or module path, or a class
                instance.

            "pool_size" : int or list, optional
                Size of the pooling window. If an `int`, all pooling layer
                will have the same pool size. If a list, the list length must
                equal "num_conv_layers". If `None` and the pooling type
                is either
                :tf_main:`MaxPooling <layers/MaxPooling1D>` or
                :tf_main:`AveragePooling <layers/AveragePooling1D>`, the
                pool size will be set to input size. That is, the output of
                the pooling layer is a single unit.

            "pool_strides" : int or list, optional
                Strides of the pooling operation. If an `int`, all pooling layer
                will have the same stride. If a list, the list length must
                equal "num_conv_layers".

            "other_pool_kwargs" : dict, optional
                Other keyword arguments for pooling layer class constructor.

        3. For **dense** layers (note that here dense layers always follow conv
           and pooling layers):

            "num_dense_layers" : int
                Number of dense layers.

            "dense_size" : int or list
                Number of units of each dense layers. If an `int`, all dense
                layers will have the same size. If a list of `int`, the list
                length must equal "num_dense_layers".

            "dense_activation" : str or callable
                Activation function applied to the output of the dense
                layers **except** the last dense layer output . Set to
                "indentity" to maintain a linear activation.
                See :func:`~texar.core.get_activation_fn` for more details.

            "dense_activation_kwargs" : dict, optional
                Keyword arguments for dense layer activation functions before
                the last dense layer.
                See :func:`~texar.core.get_activation_fn` for more details.

            "final_dense_activation" : str or callable
                Activation function applied to the output of the **last** dense
                layer. Set to `None` or
                "indentity" to maintain a linear activation.
                See :func:`~texar.core.get_activation_fn` for more details.

            "final_dense_activation_kwargs" : dict, optional
                Keyword arguments for the activation function of last
                dense layer.
                See :func:`~texar.core.get_activation_fn` for more details.

            "other_dense_kwargs" : dict, optional
                Other keyword arguments for
                :tf_main:`Dense <layers/Dense>`
                layer class constructor.

        4. For **dropouts**:

            "dropout_conv" : int or list
                The indexes of conv layers (starting from `0`) whose **inputs**
                are applied with dropout. The index = :attr:`num_conv_layers`
                means dropout applies to the final conv layer output. E.g.,

                .. code-block:: python

                    {
                        "num_conv_layers": 2,
                        "dropout_conv": [0, 2]
                    }

                will leads to a series of layers as
                `-dropout-conv0-conv1-dropout-`.

                The dropout mode (training or not) is controlled
                by the :attr:`mode` argument of :meth:`_build`.

            "dropout_dense" : int or list
                Same as "dropout_conv" but applied to dense layers (index
                starting from `0`).

            "dropout_rate" : float
                The dropout rate, between 0 and 1. E.g.,
                `"dropout_rate": 0.1` would drop out 10% of elements.

        5. Others:

            "name" : str
                Name of the network.
        """
        return {
            # Conv layers
            "num_conv_layers": 1,
            "filters": 128,
            "kernel_size": [3, 4, 5],
            "conv_activation": "relu",
            "conv_activation_kwargs": None,
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
            "dense_activation_kwargs": None,
            "final_dense_activation": None,
            "final_dense_activation_kwargs": None,
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
        activation_fn = get_activation_fn(
            self._hparams.conv_activation,
            self._hparams.conv_activation_kwargs)
        for i in range(nconv):
            hparams_i = []
            names = []
            for ks_ij in kernel_size[i]:
                name = uniquify_str("conv_%d" % (i+1), names)
                names.append(name)
                conv_kwargs_ij = {
                    "filters": filters[i],
                    "kernel_size": ks_ij,
                    "activation": activation_fn,
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
        activation_fn = get_activation_fn(
            self._hparams.dense_activation,
            self._hparams.dense_activation_kwargs)
        for i in range(ndense):
            if i == ndense - 1:
                activation_fn = get_activation_fn(
                    self._hparams.final_dense_activation,
                    self._hparams.final_dense_activation_kwargs)

            kwargs_i = {"units": dense_size[i],
                        "activation": activation_fn,
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

    def _build(self,    # pylint: disable=arguments-differ
               inputs,
               sequence_length=None,
               dtype=None,
               mode=None):
        """Feeds forward inputs through the network layers and returns outputs.

        Args:
            inputs: The inputs to the network, which is a 3D tensor.
            sequence_length (optional): An int tensor of shape `[batch_size]`
                containing the length of each element in :attr:`inputs`.
                If given, time steps beyond the length will first be masked out
                before feeding to the layers.
            dtype (optional): Type of the inputs. If not provided, infers
                from inputs automatically.
            mode (optional): A tensor taking value in
                :tf_main:`tf.estimator.ModeKeys <estimator/ModeKeys>`, including
                `TRAIN`, `EVAL`, and `PREDICT`. If `None`,
                :func:`texar.global_mode` is used.

        Returns:
            The output of the final layer.
        """
        if sequence_length is not None:
            inputs = mask_sequences(
                inputs, sequence_length, dtype=dtype, time_major=False,
                tensor_rank=3)
        return super(Conv1DNetwork, self)._build(inputs, mode=mode)

