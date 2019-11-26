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
Various neural network layers
"""

import tensorflow as tf

from texar.tf.hyperparams import HParams
from texar.tf.utils import utils
from texar.tf.utils.dtypes import is_str


__all__ = [
    "default_regularizer_hparams",
    "get_regularizer",
    "get_initializer",
    "get_activation_fn",
    "get_constraint_fn",
    "get_layer",
]


def default_regularizer_hparams():
    r"""Returns the hyperparameters and their default values of a variable
    regularizer:

    .. code-block:: python

        {
            "type": "L1L2",
            "kwargs": {
                "l1": 0.,
                "l2": 0.
            }
        }

    The default value corresponds to :tf_main:`L1L2 <keras/regularizers/L1L2>`
    and, with ``(l1=0, l2=0)``, disables regularization.
    """
    return {
        "type": "L1L2",
        "kwargs": {
            "l1": 0.,
            "l2": 0.
        }
    }


def get_regularizer(hparams=None):
    r"""Returns a variable regularizer instance.

    See :func:`~texar.tf.core.default_regularizer_hparams` for all
    hyperparameters and default values.

    The "type" field can be a subclass
    of :tf_main:`Regularizer <keras/regularizers/Regularizer>`, its string name
    or module path, or a class instance.

    Args:
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparameters are set to default values.

    Returns:
        A :tf_main:`Regularizer <keras/regularizers/Regularizer>` instance.
        `None` if :attr:`hparams` is `None` or taking the default
        hyperparameter value.

    Raises:
        ValueError: The resulting regularizer is not an instance of
            :tf_main:`Regularizer <keras/regularizers/Regularizer>`.
    """
    if hparams is None:
        return None

    if isinstance(hparams, dict):
        hparams = HParams(hparams, default_regularizer_hparams())

    rgl = utils.check_or_get_instance(
        hparams.type, hparams.kwargs.todict(),
        ["tensorflow.keras.regularizers", "texar.tf.custom"])

    if not isinstance(rgl, tf.keras.regularizers.Regularizer):
        raise ValueError("The regularizer must be an instance of "
                         "tf.keras.regularizers.Regularizer.")

    if isinstance(rgl, tf.keras.regularizers.L1L2) and \
            rgl.l1 == 0. and rgl.l2 == 0.:
        return None

    return rgl


def get_initializer(hparams=None):
    r"""Returns an initializer instance.

    Args:
        hparams (dict or HParams, optional): Hyperparameters with the structure

            .. code-block:: python

                {
                    "type": "initializer_class_or_function",
                    "kwargs": {
                        # ...
                    }
                }

            The `"type"` field can be a initializer class, its name or module
            path, or class instance. If class name is provided, it must
            be from one of the following modules:
            :tf_main:`tf.initializers <initializers>`,
            :tf_main:`tf.keras.initializers <keras/initializers>`,
            :tf_main:`tf < >`, and :mod:`texar.tf.custom`. The class is created
            by :python:`initializer_class(**kwargs)`. If a class instance
            is given, `"kwargs"` is ignored and can be omitted.

            Besides, the `"type"` field can also be an initialization function
            called with :python:`initialization_fn(**kwargs)`. In this case
            `"type"` can be the function, or its name or module path. If
            function name is provided, the function must be from one of the
            above modules or module `tfa.layers`.If no
            keyword argument is required, `"kwargs"` can be omitted.

    Returns:
        An initializer instance. `None` if :attr:`hparams` is `None`.
    """
    if hparams is None:
        return None

    kwargs = hparams.get("kwargs", {})
    if isinstance(kwargs, HParams):
        kwargs = kwargs.todict()
    modules = ["tensorflow.initializers", "tensorflow.keras.initializers",
               "tensorflow", "texar.tf.custom"]
    try:
        initializer = utils.check_or_get_instance(hparams["type"], kwargs,
                                                  modules)
    except (TypeError, ValueError):
        modules = ['tensorflow_addons.layers'] + modules
        initializer_fn = utils.get_function(hparams["type"], modules)
        initializer = initializer_fn(**kwargs)

    return initializer


def get_activation_fn(fn_name="identity", kwargs=None):
    r"""Returns an activation function `fn` with the signature
    `output = fn(input)`.

    If the function specified by :attr:`fn_name` has more than one arguments
    without default values, then all these arguments except the input feature
    argument must be specified in :attr:`kwargs`. Arguments with default values
    can also be specified in :attr:`kwargs` to take values other than the
    defaults. In this case a partial function is returned with the above
    signature.

    Args:
        fn_name (str or callable): An activation function, or its name or
            module path. The function can be:

            - Built-in function defined in :tf_main:`tf < >` or
              :tf_main:`tf.nn <nn>`, e.g., :tf_main:`tf.identity <identity>`.
            - User-defined activation functions in module
              :mod:`texar.tf.custom`.
            - External activation functions. Must provide the full module path,
              e.g., ``"my_module.my_activation_fn"``.

        kwargs (optional): A `dict` or instance of :class:`~texar.tf.HParams`
            containing the keyword arguments of the activation function.

    Returns:
        An activation function. `None` if :attr:`fn_name` is `None`.
    """
    if fn_name is None:
        return None

    fn_modules = ['tensorflow', 'tensorflow.nn', 'texar.tf.custom',
                  'texar.tf.core.layers']
    activation_fn_ = utils.get_function(fn_name, fn_modules)
    activation_fn = activation_fn_

    # Make a partial function if necessary
    if kwargs is not None:
        if isinstance(kwargs, HParams):
            kwargs = kwargs.todict()

        def _partial_fn(features):
            return activation_fn_(features, **kwargs)
        activation_fn = _partial_fn

    return activation_fn


def get_constraint_fn(fn_name="NonNeg"):
    r"""Returns a constraint function.

    The function must follow the signature: :python:`w_ = constraint_fn(w)`.

    Args:
        fn_name (str or callable): The name or full path to a
            constraint function, or the function itself.

            The function can be:

            - Built-in constraint functions defined in modules
              :tf_main:`tf.keras.constraints <keras/constraints>`
              (e.g., :tf_main:`NonNeg <keras/constraints/NonNeg>`)
              or :tf_main:`tf < >` or :tf_main:`tf.nn <nn>`
              (e.g., activation functions).
            - User-defined function in :mod:`texar.tf.custom`.
            - Externally defined function. Must provide the full path,
              e.g., ``"my_module.my_constraint_fn"``.

            If a callable is provided, then it is returned directly.

    Returns:
        The constraint function. `None` if :attr:`fn_name` is `None`.
    """
    if fn_name is None:
        return None

    fn_modules = ['tensorflow.keras.constraints', 'tensorflow',
                  'tensorflow.nn', 'texar.tf.custom']
    constraint_fn = utils.get_function(fn_name, fn_modules)
    return constraint_fn


def get_layer(hparams):
    r"""Makes a layer instance.

    The layer must be an instance of :tf_main:`tf.keras.layers.Layer`.

    Args:
        hparams (dict or HParams): Hyperparameters of the layer, with structure:

            .. code-block:: python

                {
                    "type": "LayerClass",
                    "kwargs": {
                        # Keyword arguments of the layer class
                        # ...
                    }
                }

            Here:

            `"type"`: str or layer class or layer instance
                The layer type. This can be

                - The string name or full module path of a layer class. If
                  the class name is provided, the class must be in module
                  :tf_main:`tf.keras.layers`, :mod:`texar.tf.core`,
                  or :mod:`texar.tf.custom`.
                - A layer class.
                - An instance of a layer class.

                For example

                .. code-block:: python

                    "type": "Conv1D"                           # class name
                    "type": "texar.tf.core.MaxReducePooling1D" # module path
                    "type": "my_module.MyLayer"                # module path
                    "type": tf.layers.Conv2D                   # class
                    "type": Conv1D(filters=10, kernel_size=2)  # cell instance
                    "type": MyLayer(...)                       # cell instance

            `"kwargs"`: dict
                A dictionary of keyword arguments for constructor of the
                layer class. Ignored if :attr:`"type"` is a layer instance.

                - Arguments named "activation" can be a callable,
                  or a `str` of the name or module path to the activation
                  function.
                - Arguments named "\*_regularizer" and "\*_initializer"
                  can be a class instance, or a `dict` of hyperparameters of
                  respective regularizers and initializers. See
                - Arguments named "\*_constraint" can be a callable, or a
                  `str` of the name or full path to the constraint function.

    Returns:
        A layer instance. If ``hparams["type"]`` is a layer instance, returns it
        directly.

    Raises:
        ValueError: If :attr:`hparams` is `None`.
        ValueError: If the resulting layer is not an instance of
            :tf_main:`tf.keras.layers.Layer`.
    """
    if hparams is None:
        raise ValueError("`hparams` must not be `None`.")

    layer_type = hparams["type"]
    if not is_str(layer_type) and not isinstance(layer_type, type):
        layer = layer_type
    else:
        layer_modules = ["tensorflow.keras.layers", "texar.tf.core",
                         "texar.tf.custom"]
        layer_class = utils.check_or_get_class(layer_type, layer_modules)
        if isinstance(hparams, dict):
            default_kwargs = _layer_class_to_default_kwargs_map.get(layer_class,
                                                                    {})
            default_hparams = {"type": layer_type, "kwargs": default_kwargs}
            hparams = HParams(hparams, default_hparams)

        kwargs = {}
        for k, v in hparams.kwargs.items():
            if k.endswith('_regularizer'):
                kwargs[k] = get_regularizer(v)
            elif k.endswith('_initializer'):
                kwargs[k] = get_initializer(v)
            elif k.endswith('activation'):
                kwargs[k] = get_activation_fn(v)
            elif k.endswith('_constraint'):
                kwargs[k] = get_constraint_fn(v)
            else:
                kwargs[k] = v
        layer = utils.get_instance(layer_type, kwargs, layer_modules)

    if not isinstance(layer, tf.keras.layers.Layer):
        raise ValueError(
            "layer must be an instance of `tf.keras.layers.Layer`.")

    return layer


def _common_default_conv_dense_kwargs():
    r"""Returns the default keyword argument values that are common to
    convolution layers.
    """
    return {
        "activation": None,
        "use_bias": True,
        "kernel_initializer": {
            "type": "glorot_uniform",
            "kwargs": {}
        },
        "bias_initializer": {
            "type": "zeros_initializer",
            "kwargs": {}
        },
        "kernel_regularizer": default_regularizer_hparams(),
        "bias_regularizer": default_regularizer_hparams(),
        "activity_regularizer": default_regularizer_hparams(),
        "kernel_constraint": None,
        "bias_constraint": None
    }


def default_conv1d_kwargs():
    r"""Returns the default keyword argument values of the constructor
    of 1D-convolution layer class
    :tf_main:`tf.keras.layers.Conv1D <layers/Conv1D>`.

    .. code-block:: python
        {
            "filters": 100,
            "kernel_size": 3,
            "strides": 1,
            "padding": 'valid',
            "data_format": 'channels_last',
            "dilation_rate": 1
            "activation": "identity",
            "use_bias": True,
            "kernel_initializer": {
                "type": "glorot_uniform",
                "kwargs": {}
            },
            "bias_initializer": {
                "type": "zeros_initializer",
                "kwargs": {}
            },
            "kernel_regularizer": {
                "type": "L1L2",
                "kwargs": {
                    "l1": 0.,
                    "l2": 0.
                }
            },
            "bias_regularizer": {
                # same as in "kernel_regularizer"
                # ...
            },
            "activity_regularizer": {
                # same as in "kernel_regularizer"
                # ...
            },
            "kernel_constraint": None,
            "bias_constraint": None,
        }
    """
    kwargs = _common_default_conv_dense_kwargs()
    kwargs.update({
        "kernel_size": 3,
        "filters": 100,
        "strides": 1,
        "dilation_rate": 1,
        "data_format": "channels_last"
    })
    return kwargs


def default_conv2d_kwargs():
    r"""TODO
    """
    return {}


def default_conv3d_kwargs():
    r"""TODO
    """
    return {}


def default_conv2d_transpose_kwargs():
    r"""TODO
    """
    return {}


def default_conv3d_transpose_kwargs():
    r"""TODO
    """
    return {}


def default_dense_kwargs():
    r"""Returns the default keyword argument values of the constructor
    of the dense layer class :tf_main:`tf.keras.layers.Dense <layers/Dense>`.

    .. code-block:: python

        {
            "units": 256,
            "activation": "identity",
            "use_bias": True,
            "kernel_initializer": {
                "type": "glorot_uniform",
                "kwargs": {}
            },
            "bias_initializer": {
                "type": "zeros_initializer",
                "kwargs": {}
            },
            "kernel_regularizer": {
                "type": "L1L2",
                "kwargs": {
                    "l1": 0.,
                    "l2": 0.
                }
            },
            "bias_regularizer": {
                # same as in "kernel_regularizer"
                # ...
            },
            "activity_regularizer": {
                # same as in "kernel_regularizer"
                # ...
            },
            "kernel_constraint": None,
            "bias_constraint": None,
        }
    """
    kwargs = _common_default_conv_dense_kwargs()
    kwargs.update({
        "units": 256
    })
    return kwargs


def default_dropout_kwargs():
    r"""TODO
    """
    return {}


def default_flatten_kwargs():
    r"""TODO
    """
    return {}


def default_max_pooling1d_kwargs():
    r"""TODO
    """
    return {}


def default_max_pooling2d_kwargs():
    r"""TODO
    """
    return {}


def default_max_pooling3d_kwargs():
    r"""TODO
    """
    return {}


def default_separable_conv2d_kwargs():
    r"""TODO
    """
    return {}


def default_batch_normalization_kwargs():
    r"""TODO
    """
    return {}


def default_average_pooling1d_kwargs():
    r"""TODO
    """
    return {}


def default_average_pooling2d_kwargs():
    r"""TODO
    """
    return {}


def default_average_pooling3d_kwargs():
    r"""TODO
    """
    return {}


_layer_class_to_default_kwargs_map = {
    tf.keras.layers.Conv1D: default_conv1d_kwargs(),
    tf.keras.layers.Conv2D: default_conv2d_kwargs(),
    tf.keras.layers.Conv3D: default_conv3d_kwargs(),
    tf.keras.layers.Conv2DTranspose: default_conv2d_transpose_kwargs(),
    tf.keras.layers.Conv3DTranspose: default_conv3d_transpose_kwargs(),
    tf.keras.layers.Dense: default_dense_kwargs(),
    tf.keras.layers.Dropout: default_dropout_kwargs(),
    tf.keras.layers.Flatten: default_flatten_kwargs(),
    tf.keras.layers.MaxPooling1D: default_max_pooling1d_kwargs(),
    tf.keras.layers.MaxPooling2D: default_max_pooling2d_kwargs(),
    tf.keras.layers.MaxPooling3D: default_max_pooling3d_kwargs(),
    tf.keras.layers.SeparableConv2D: default_separable_conv2d_kwargs(),
    tf.keras.layers.BatchNormalization: default_batch_normalization_kwargs(),
    tf.keras.layers.AveragePooling1D: default_average_pooling1d_kwargs(),
    tf.keras.layers.AveragePooling2D: default_average_pooling2d_kwargs(),
    tf.keras.layers.AveragePooling3D: default_average_pooling3d_kwargs(),
}
