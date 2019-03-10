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
Various neural network layers
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import copy

import tensorflow as tf
import tensorflow.contrib.rnn as rnn

from texar.hyperparams import HParams
from texar.utils import utils
from texar.utils.dtypes import is_str
from texar.utils.variables import add_variable
from texar.utils.mode import is_train_mode, switch_dropout

# pylint: disable=redefined-variable-type, invalid-name
# pylint: disable=too-many-branches, too-many-arguments, too-many-lines
# pylint: disable=protected-access

__all__ = [
    "default_rnn_cell_hparams",
    "get_rnn_cell",
    "get_rnn_cell_trainable_variables",
    "default_regularizer_hparams",
    "get_regularizer",
    "get_initializer",
    "get_activation_fn",
    "get_constraint_fn",
    "get_layer",
    "_ReducePooling1D",
    "MaxReducePooling1D",
    "AverageReducePooling1D",
    "get_pooling_layer_hparams",
    "MergeLayer",
    "SequentialLayer",
    "default_conv1d_kwargs",
    "default_conv2d_kwargs",
    "default_conv3d_kwargs",
    "default_conv2d_transpose_kwargs",
    "default_conv3d_transpose_kwargs",
    "default_dense_kwargs",
    "default_dropout_kwargs",
    "default_flatten_kwargs",
    "default_max_pooling1d_kwargs",
    "default_max_pooling2d_kwargs",
    "default_max_pooling3d_kwargs",
    "default_separable_conv2d_kwargs",
    "default_batch_normalization_kwargs",
    "default_average_pooling1d_kwargs",
    "default_average_pooling2d_kwargs",
    "default_average_pooling3d_kwargs",
    "layer_normalize",
]

def default_rnn_cell_hparams():
    """Returns a `dict` of RNN cell hyperparameters and their default values.

    .. role:: python(code)
       :language: python

    .. code-block:: python

        {
            "type": "LSTMCell",
            "kwargs": {
                "num_units": 256
            },
            "num_layers": 1,
            "dropout": {
                "input_keep_prob": 1.0,
                "output_keep_prob": 1.0,
                "state_keep_prob": 1.0,
                "variational_recurrent": False,
                "input_size": []
            },
            "residual": False,
            "highway": False,
        }

    Here:

    "type" : str or cell class or cell instance
        The RNN cell type. This can be

        - The string name or full module path of a cell class. If class \
        name is provided, the class must be in module \
        :tf_main:`tf.nn.rnn_cell <nn/rnn_cell/LSTMCell>`, \
        :tf_main:`tf.contrib.rnn <contrib/rnn>`, or :mod:`texar.custom`.
        - A cell class.
        - An instance of a cell class. This is not valid if \
        "num_layers" > 1.

        For example

        .. code-block:: python

            "type": "LSTMCell" # class name
            "type": "tensorflow.contrib.rnn.Conv1DLSTMCell" # module path
            "type": "my_module.MyCell" # module path
            "type": tf.nn.rnn_cell.GRUCell # class
            "type": BasicRNNCell(num_units=100) # cell instance
            "type": MyCell(...) # cell instance

    "kwargs" : dict
        Keyword arguments for the constructor of the cell class.
        A cell is created by :python:`cell_class(**kwargs)`, where
        `cell_class` is specified in "type" above.

        Ignored if "type" is a cell instance.

    "num_layers" : int
        Number of cell layers. Each layer is a cell created as above, with
        the same hyperparameters specified in "kwargs".

    "dropout" : dict
        Dropout applied to the cell in **each** layer. See
        :tf_main:`DropoutWrapper <contrib/rnn/DropoutWrapper>` for details of
        the hyperparameters. If all "\*_keep_prob" = 1, no dropout is applied.

        Specifically, if "variational_recurrent" = `True`,
        the same dropout mask is applied across all time steps per run call.
        If `True`, "input_size" is required, which is a list of input
        size of each cell layer. The input size of a cell layer is the last
        dimension size of its input tensor. For example, the
        input size of the first layer is usually the dimension of
        word embeddings, while the input size of subsequent layers
        are usually the `num_units` of the preceding-layer cell. E.g.,

        .. code-block:: python

            # Assume embedding_dim = 100
            "type": "LSTMCell",
            "kwargs": { "num_units": 123 },
            "num_layers": 3,
            "dropout": {
                "output_keep_prob": 0.5,
                "variational_recurrent": True,
                "input_size": [100, 123, 123]
            }

    "residual" : bool
        If `True`, apply residual connection on the inputs and
        outputs of cell in **each** layer except the first layer. Ignored
        if "num_layers" = 1.

    "highway" : bool
        If True, apply highway connection on the inputs and
        outputs of cell in each layer except the first layer. Ignored if
        "num_layers" = 1.
    """
    return {
        "type": "LSTMCell",
        "kwargs": {
            "num_units": 256,
        },
        "num_layers": 1,
        "dropout": {
            "input_keep_prob": 1.0,
            "output_keep_prob": 1.0,
            "state_keep_prob": 1.0,
            "variational_recurrent": False,
            "input_size": [],
            "@no_typecheck": [
                "input_keep_prob", "output_keep_prob", "state_keep_prob"
            ]
        },
        "residual": False,
        "highway": False,
        "@no_typecheck": ["type"]
    }

def get_rnn_cell(hparams=None, mode=None):
    """Creates an RNN cell.

    See :func:`~texar.core.default_rnn_cell_hparams` for all
    hyperparameters and default values.

    Args:
        hparams (dict or HParams, optional): Cell hyperparameters. Missing
            hyperparameters are set to default values.
        mode (optional): A Tensor taking value in
            :tf_main:`tf.estimator.ModeKeys <estimator/ModeKeys>`, including
            `TRAIN`, `EVAL`, and `PREDICT`. If `None`, dropout will be
            controlled by :func:`texar.global_mode`.

    Returns:
        A cell instance.

    Raises:
        ValueError: If hparams["num_layers"]>1 and hparams["type"] is a class
            instance.
        ValueError: The cell is not an
            :tf_main:`RNNCell <contrib/rnn/RNNCell>` instance.
    """
    if hparams is None or isinstance(hparams, dict):
        hparams = HParams(hparams, default_rnn_cell_hparams())

    d_hp = hparams["dropout"]
    if d_hp["variational_recurrent"] and \
            len(d_hp["input_size"]) != hparams["num_layers"]:
        raise ValueError(
            "If variational_recurrent=True, input_size must be a list of "
            "num_layers(%d) integers. Got len(input_size)=%d." %
            (hparams["num_layers"], len(d_hp["input_size"])))

    cells = []
    cell_kwargs = hparams["kwargs"].todict()
    num_layers = hparams["num_layers"]
    for layer_i in range(num_layers):
        # Create the basic cell
        cell_type = hparams["type"]
        if not is_str(cell_type) and not isinstance(cell_type, type):
            if num_layers > 1:
                raise ValueError(
                    "If 'num_layers'>1, then 'type' must be a cell class or "
                    "its name/module path, rather than a cell instance.")
        cell_modules = ['tensorflow.nn.rnn_cell', 'tensorflow.contrib.rnn',
                        'texar.custom']
        cell = utils.check_or_get_instance(
            cell_type, cell_kwargs, cell_modules, rnn.RNNCell)

        # Optionally add dropout
        if d_hp["input_keep_prob"] < 1.0 or \
                d_hp["output_keep_prob"] < 1.0 or \
                d_hp["state_keep_prob"] < 1.0:
            vr_kwargs = {}
            if d_hp["variational_recurrent"]:
                vr_kwargs = {
                    "variational_recurrent": True,
                    "input_size": d_hp["input_size"][layer_i],
                    "dtype": tf.float32
                }
            input_keep_prob = switch_dropout(d_hp["input_keep_prob"],
                                             mode)
            output_keep_prob = switch_dropout(d_hp["output_keep_prob"],
                                              mode)
            state_keep_prob = switch_dropout(d_hp["state_keep_prob"],
                                             mode)
            cell = rnn.DropoutWrapper(
                cell=cell,
                input_keep_prob=input_keep_prob,
                output_keep_prob=output_keep_prob,
                state_keep_prob=state_keep_prob,
                **vr_kwargs)

        # Optionally add residual and highway connections
        if layer_i > 0:
            if hparams["residual"]:
                cell = rnn.ResidualWrapper(cell)
            if hparams["highway"]:
                cell = rnn.HighwayWrapper(cell)

        cells.append(cell)

    if hparams["num_layers"] > 1:
        cell = rnn.MultiRNNCell(cells)
    else:
        cell = cells[0]

    return cell

def get_rnn_cell_trainable_variables(cell):
    """Returns the list of trainable variables of an RNN cell.

    Args:
        cell: an instance of :tf_main:`RNNCell <nn/rnn_cell/RNNCell>`.

    Returns:
        list: trainable variables of the cell.
    """
    cell_ = cell
    while True:
        try:
            return cell_.trainable_variables
        except AttributeError:
        # Cell wrappers (e.g., `DropoutWrapper`) cannot directly access to
        # `trainable_variables` as they don't initialize superclass
        # (tf==v1.3). So try to access through the cell in the wrapper.
            cell_ = cell._cell  # pylint: disable=protected-access

def default_regularizer_hparams():
    """Returns the hyperparameters and their default values of a variable
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
    and, with `(l1=0, l2=0)`, disables regularization.
    """
    return {
        "type": "L1L2",
        "kwargs": {
            "l1": 0.,
            "l2": 0.
        }
    }

def get_regularizer(hparams=None):
    """Returns a variable regularizer instance.

    See :func:`~texar.core.default_regularizer_hparams` for all
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
        ["tensorflow.keras.regularizers", "texar.custom"])

    if not isinstance(rgl, tf.keras.regularizers.Regularizer):
        raise ValueError("The regularizer must be an instance of "
                         "tf.keras.regularizers.Regularizer.")

    if isinstance(rgl, tf.keras.regularizers.L1L2) and \
            rgl.l1 == 0. and rgl.l2 == 0.:
        return None

    return rgl

def get_initializer(hparams=None):
    """Returns an initializer instance.

    .. role:: python(code)
       :language: python

    Args:
        hparams (dict or HParams, optional): Hyperparameters with the structure

            .. code-block:: python

                {
                    "type": "initializer_class_or_function",
                    "kwargs": {
                        #...
                    }
                }

            The "type" field can be a initializer class, its name or module
            path, or class instance. If class name is provided, the class must
            be from one the following modules:
            :tf_main:`tf.initializers <initializers>`,
            :tf_main:`tf.keras.initializers <keras/initializers>`,
            :tf_main:`tf < >`, and :mod:`texar.custom`. The class is created
            by :python:`initializer_class(**kwargs)`. If a class instance
            is given, "kwargs" is ignored and can be omitted.

            Besides, the "type" field can also be an initialization function
            called with :python:`initialization_fn(**kwargs)`. In this case
            "type" can be the function, or its name or module path. If
            function name is provided, the function must be from one of the
            above modules or module `tf.contrib.layers`. If no
            keyword argument is required, "kwargs" can be omitted.

    Returns:
        An initializer instance. `None` if :attr:`hparams` is `None`.
    """
    if hparams is None:
        return None

    kwargs = hparams.get("kwargs", {})
    if isinstance(kwargs, HParams):
        kwargs = kwargs.todict()
    modules = ["tensorflow.initializers", "tensorflow.keras.initializers",
               "tensorflow", "texar.custom"]
    try:
        initializer = utils.check_or_get_instance(hparams["type"], kwargs,
                                                  modules)
    except TypeError:
        modules += ['tensorflow.contrib.layers']
        initializer_fn = utils.get_function(hparams["type"], modules)
        initializer = initializer_fn(**kwargs)

    return initializer

def get_activation_fn(fn_name="identity", kwargs=None):
    """Returns an activation function `fn` with the signature
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

            - Built-in function defined in :tf_main:`tf < >` or \
            :tf_main:`tf.nn <nn>`, e.g., :tf_main:`tf.identity <identity>`.
            - User-defined activation functions in module :mod:`texar.custom`.
            - External activation functions. Must provide the full module path,\
              e.g., "my_module.my_activation_fn".

        kwargs (optional): A `dict` or instance of :class:`~texar.HParams`
            containing the keyword arguments of the activation function.

    Returns:
        An activation function. `None` if :attr:`fn_name` is `None`.
    """
    if fn_name is None:
        return None

    fn_modules = ['tensorflow', 'tensorflow.nn', 'texar.custom',
                  'texar.core.layers']
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
    """Returns a constraint function.

    .. role:: python(code)
       :language: python

    The function must follow the signature:
    :python:`w_ = constraint_fn(w)`.

    Args:
        fn_name (str or callable): The name or full path to a
            constraint function, or the function itself.

            The function can be:

            - Built-in constraint functions defined in modules \
            :tf_main:`tf.keras.constraints <keras/constraints>` \
            (e.g., :tf_main:`NonNeg <keras/constraints/NonNeg>`) \
            or :tf_main:`tf < >` or :tf_main:`tf.nn <nn>` \
            (e.g., activation functions).
            - User-defined function in :mod:`texar.custom`.
            - Externally defined function. Must provide the full path, \
            e.g., `"my_module.my_constraint_fn"`.

            If a callable is provided, then it is returned directly.

    Returns:
        The constraint function. `None` if :attr:`fn_name` is `None`.
    """
    if fn_name is None:
        return None

    fn_modules = ['tensorflow.keras.constraints', 'tensorflow',
                  'tensorflow.nn', 'texar.custom']
    constraint_fn = utils.get_function(fn_name, fn_modules)
    return constraint_fn

def get_layer(hparams):
    """Makes a layer instance.

    The layer must be an instance of :tf_main:`tf.layers.Layer <layers/Layer>`.

    Args:
        hparams (dict or HParams): Hyperparameters of the layer, with
            structure:

            .. code-block:: python

                {
                    "type": "LayerClass",
                    "kwargs": {
                        # Keyword arguments of the layer class
                        # ...
                    }
                }

            Here:

            "type" : str or layer class or layer instance
                The layer type. This can be

                - The string name or full module path of a layer class. If \
                the class name is provided, the class must be in module \
                :tf_main:`tf.layers <layers>`, :mod:`texar.core`, \
                or :mod:`texar.custom`.
                - A layer class.
                - An instance of a layer class.

                For example

                .. code-block:: python

                    "type": "Conv1D" # class name
                    "type": "texar.core.MaxReducePooling1D" # module path
                    "type": "my_module.MyLayer" # module path
                    "type": tf.layers.Conv2D # class
                    "type": Conv1D(filters=10, kernel_size=2) # cell instance
                    "type": MyLayer(...) # cell instance

            "kwargs" : dict
                A dictionary of keyword arguments for constructor of the
                layer class. Ignored if :attr:`"type"` is a layer instance.

                - Arguments named "activation" can be a callable, \
                or a `str` of \
                the name or module path to the activation function.
                - Arguments named "\*_regularizer" and "\*_initializer" \
                can be a class instance, or a `dict` of \
                hyperparameters of \
                respective regularizers and initializers. See
                - Arguments named "\*_constraint" can be a callable, or a \
                `str` of the name or full path to the constraint function.

    Returns:
        A layer instance. If hparams["type"] is a layer instance, returns it
        directly.

    Raises:
        ValueError: If :attr:`hparams` is `None`.
        ValueError: If the resulting layer is not an instance of
            :tf_main:`tf.layers.Layer <layers/Layer>`.
    """
    if hparams is None:
        raise ValueError("`hparams` must not be `None`.")

    layer_type = hparams["type"]
    if not is_str(layer_type) and not isinstance(layer_type, type):
        layer = layer_type
    else:
        layer_modules = ["tensorflow.layers", "texar.core", "texar.costum"]
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

    if not isinstance(layer, tf.layers.Layer):
        raise ValueError("layer must be an instance of `tf.layers.Layer`.")

    return layer


def _compute_concat_output_shape(input_shape, axis):
    """Infers the output shape of concat given the input shape.

    The code is adapted from the ConcatLayer of lasagne
    (https://github.com/Lasagne/Lasagne/blob/master/lasagne/layers/merge.py)

    Args:
        input_shape (list): A list of shapes, each of which is in turn a
            list or TensorShape.
        axis (int): Axis of the concat operation.

    Returns:
        list: Output shape of concat.
    """
    # The size of each axis of the output shape equals the first
    # input size of respective axis that is not `None`
    input_shape = [tf.TensorShape(s).as_list() for s in input_shape]
    output_shape = [next((s for s in sizes if s is not None), None)
                    for sizes in zip(*input_shape)]
    axis_sizes = [s[axis] for s in input_shape]
    concat_axis_size = None if any(s is None for s in axis_sizes) \
            else sum(axis_sizes)
    output_shape[axis] = concat_axis_size
    return output_shape

class _ReducePooling1D(tf.layers.Layer):
    """Pooling layer for arbitrary reduce functions for 1D inputs.

    The same as `tf.python.layers.pooling._Pooling1D` except that the pooling
    dimension is entirely reduced (i.e., `pool_size=length`).

    This class is for code reuse, rather than an exposed API.
    """
    def __init__(self, reduce_function, data_format='channels_last',
                 name=None, **kwargs):
        super(_ReducePooling1D, self).__init__(name=name, **kwargs)
        self._reduce_function = reduce_function
        if data_format not in {'channels_last', 'channels_first'}:
            raise ValueError("`data_format must be either 'channels_last' or` "
                             "'channels_first'. Got: {}".format(data_format))
        self._data_format = data_format

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        if self._data_format == 'channels_last':
            return tf.TensorShape([input_shape[0], input_shape[2]])
        else:
            return tf.TensorShape([input_shape[0], input_shape[1]])

    def call(self, inputs):
        if self._data_format == 'channels_last':
            return self._reduce_function(inputs, axis=1)
        else:
            return self._reduce_function(inputs, axis=2)

class MaxReducePooling1D(_ReducePooling1D):
    """A subclass of :tf_main:`tf.layers.Layer <layers/Layer>`.
    Max Pooling layer for 1D inputs. The same as
    :tf_main:`MaxPooling1D <layers/MaxPooling1D>` except that the pooling
    dimension is entirely reduced (i.e., `pool_size=input_length`).
    """
    def __init__(self, data_format='channels_last', name=None, **kwargs):
        super(MaxReducePooling1D, self).__init__(
            tf.reduce_max, data_format=data_format, name=name, **kwargs)

class AverageReducePooling1D(_ReducePooling1D):
    """A subclass of :tf_main:`tf.layers.Layer <layers/Layer>`.
    Average Pooling layer for 1D inputs. The same as
    :tf_main:`AveragePooling1D <layers/AveragePooling1D>` except that the
    pooling dimension is entirely reduced (i.e., `pool_size=input_length`).
    """
    def __init__(self, data_format='channels_last', name=None, **kwargs):
        super(AverageReducePooling1D, self).__init__(
            tf.reduce_mean, data_format=data_format, name=name, **kwargs)

_POOLING_TO_REDUCE = {
    "MaxPooling1D": "MaxReducePooling1D",
    "AveragePooling1D": "AverageReducePooling1D",
    tf.layers.MaxPooling1D: MaxReducePooling1D,
    tf.layers.AveragePooling1D: AverageReducePooling1D
}

def get_pooling_layer_hparams(hparams):
    """Creates pooling layer hparams `dict` usable for :func:`get_layer`.

    If the :attr:`hparams` sets `'pool_size'` to `None`, the layer will be
    changed to the respective reduce-pooling layer. For example,
    :class:`tf.layers.MaxPooling1D <layers/MaxPooling1D>` is replaced with
    :class:`~texar.core.MaxReducePooling1D`.
    """
    if isinstance(hparams, HParams):
        hparams = hparams.todict()

    new_hparams = copy.copy(hparams)
    kwargs = new_hparams.get('kwargs', None)

    if kwargs and kwargs.get('pool_size', None) is None:
        pool_type = hparams['type']
        new_hparams['type'] = _POOLING_TO_REDUCE.get(pool_type, pool_type)
        kwargs.pop('pool_size', None)
        kwargs.pop('strides', None)
        kwargs.pop('padding', None)

    return new_hparams

class MergeLayer(tf.layers.Layer):
    """A subclass of :tf_main:`tf.layers.Layer <layers/Layer>`.
    A layer that consists of multiple layers in parallel. Input is fed to
    each of the parallel layers, and the outputs are merged with a
    specified mode.

    Args:
        layers (list, optional): A list of :tf_main:`tf.layers.Layer
            <layers/layer>` instances, or a list of hyperparameter dicts
            each of which specifies type and kwargs of each layer (see
            the `hparams` argument of :func:`get_layer`).

            If `None`, this layer degenerates to a merging operator that merges
            inputs directly.
        mode (str): Mode of the merge op. This can be:

            - :attr:`'concat'`: Concatenates layer outputs along one axis. \
              Tensors must have the same shape except for the dimension \
              specified in `axis`, which can have different sizes.
            - :attr:`'elemwise_sum'`: Outputs element-wise sum.
            - :attr:`'elemwise_mul'`: Outputs element-wise product.
            - :attr:`'sum'`: Computes the sum of layer outputs along the \
              dimension given by `axis`. E.g., given `axis=1`, \
              two tensors of shape `[a, b]` and `[a, c]` respectively \
              will result in a merged tensor of shape `[a]`.
            - :attr:`'mean'`: Computes the mean of layer outputs along the \
              dimension given in `axis`.
            - :attr:`'prod'`: Computes the product of layer outputs along the \
              dimension given in `axis`.
            - :attr:`'max'`: Computes the maximum of layer outputs along the \
              dimension given in `axis`.
            - :attr:`'min'`: Computes the minimum of layer outputs along the \
              dimension given in `axis`.
            - :attr:`'and'`: Computes the `logical and` of layer outputs along \
              the dimension given in `axis`.
            - :attr:`'or'`: Computes the `logical or` of layer outputs along \
              the dimension given in `axis`.
            - :attr:`'logsumexp'`: Computes \
              log(sum(exp(elements across the dimension of layer outputs)))
        axis (int): The axis to use in merging. Ignored in modes
            :attr:`'elemwise_sum'` and :attr:`'elemwise_mul'`.
        trainable (bool): Whether the layer should be trained.
        name (str, optional): Name of the layer.
    """

    def __init__(self,
                 layers=None,
                 mode='concat',
                 axis=1,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(MergeLayer, self).__init__(
            trainable=trainable, name=name, **kwargs)
        self._mode = mode
        self._axis = axis

        self._layers = None
        if layers is not None:
            if len(layers) == 0:
                raise ValueError(
                    "'layers' must be either None or a non-empty list.")
            self._layers = []
            for layer in layers:
                if isinstance(layer, tf.layers.Layer):
                    self._layers.append(layer)
                else:
                    self._layers.append(get_layer(hparams=layer))

        # Keep tracks of whether trainable variables have been created
        self._vars_built = False

    def compute_output_shape(self, input_shape):
        if self._layers is None:
            _shapes = input_shape
            if not isinstance(_shapes, (list, tuple)):
                _shapes = [_shapes]
        else:
            _shapes = []
            for layer in self._layers:
                layer_output_shape = layer.compute_output_shape(input_shape)
                _shapes.append(layer_output_shape)
        _shapes = [tf.TensorShape(s) for s in _shapes]

        if self._mode == 'concat':
            output_shape = _compute_concat_output_shape(_shapes, self._axis)
        elif self._mode in ['sum', 'mean', 'prod', 'max', 'min',
                            'and', 'or', 'logsumexp']:
            output_shape = _compute_concat_output_shape(_shapes, self._axis)
            output_shape.pop(self._axis)
        elif self._mode in ['elemwise_sum', 'elemwise_mul']:
            # Simply infer the output shape as the input shape of highest rank
            _ranks = [s.ndims for s in _shapes]
            max_rank = max(_ranks)
            max_ranked_shapes = []
            for i, s in enumerate(_shapes):
                if _ranks[i] == max_rank:
                    max_ranked_shapes.append(s.as_list())
            # Grab the first size of each axis that is not `None`
            output_shape = [next((s for s in sizes if s is not None), None)
                            for sizes in zip(*max_ranked_shapes)]
        else:
            raise ValueError("Unknown merge mode: '%s'" % self._mode)

        return tf.TensorShape(output_shape)

    def _collect_weights(self):
        """Collects (non-)trainable weights of each of the parallel layers.
        """
        if self._layers is None:
            pass
        for layer in self._layers:
            if self.trainable:
                add_variable(
                    layer._trainable_weights, self._trainable_weights)
            else:
                add_variable(
                    layer._trainable_weights, self._non_trainable_weights)
            add_variable(
                layer._non_trainable_weights, self._non_trainable_weights)

    @property
    def trainable_weights(self):
        return self._trainable_weights

    @property
    def non_trainable_weights(self):
        return self._non_trainable_weights

    def call(self, inputs):
        if self._layers is None:
            layer_outputs = inputs
            if not isinstance(layer_outputs, (list, tuple)):
                layer_outputs = [layer_outputs]
        else:
            layer_outputs = []
            for layer in self._layers:
                layer_output = layer(inputs)
                layer_outputs.append(layer_output)

        if self._mode == 'concat':
            outputs = tf.concat(values=layer_outputs, axis=self._axis)
        elif self._mode == 'elemwise_sum':
            outputs = layer_outputs[0]
            for i in range(1, len(layer_outputs)):
                outputs = tf.add(outputs, layer_outputs[i])
        elif self._mode == 'elemwise_mul':
            outputs = layer_outputs[0]
            for i in range(1, len(layer_outputs)):
                outputs = tf.multiply(outputs, layer_outputs[i])
        elif self._mode == 'sum':
            _concat = tf.concat(values=layer_outputs, axis=self._axis)
            outputs = tf.reduce_sum(_concat, axis=self._axis)
        elif self._mode == 'mean':
            _concat = tf.concat(values=layer_outputs, axis=self._axis)
            outputs = tf.reduce_mean(_concat, axis=self._axis)
        elif self._mode == 'prod':
            _concat = tf.concat(values=layer_outputs, axis=self._axis)
            outputs = tf.reduce_prod(_concat, axis=self._axis)
        elif self._mode == 'max':
            _concat = tf.concat(values=layer_outputs, axis=self._axis)
            outputs = tf.reduce_max(_concat, axis=self._axis)
        elif self._mode == 'min':
            _concat = tf.concat(values=layer_outputs, axis=self._axis)
            outputs = tf.reduce_min(_concat, axis=self._axis)
        elif self._mode == 'and':
            _concat = tf.concat(values=layer_outputs, axis=self._axis)
            outputs = tf.reduce_all(_concat, axis=self._axis)
        elif self._mode == 'or':
            _concat = tf.concat(values=layer_outputs, axis=self._axis)
            outputs = tf.reduce_any(_concat, axis=self._axis)
        elif self._mode == 'logsumexp':
            _concat = tf.concat(values=layer_outputs, axis=self._axis)
            outputs = tf.reduce_logsumexp(_concat, axis=self._axis)
        else:
            raise ValueError("Unknown merge mode: '%s'" % self._mode)

        if not self.built or not self._vars_built:
            self._collect_weights()
            self._vars_built = True

        return outputs

    @property
    def layers(self):
        """The list of parallel layers.
        """
        return self._layers


class SequentialLayer(tf.layers.Layer):
    """A subclass of :tf_main:`tf.layers.Layer <layers/Layer>`.
    A layer that consists of multiple layers connected sequentially.

    Args:
        layers (list): A list of :tf_main:`tf.layers.Layer
            <layers/layer>` instances, or a list of hyperparameter dicts
            each of which specifying type and kwargs of each layer (see
            the `hparams` argument of :func:`get_layer`). The layers are
            connected sequentially.
    """
    def __init__(self,
                 layers,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(SequentialLayer, self).__init__(
            trainable=trainable, name=name, **kwargs)

        if len(layers) == 0:
            raise ValueError("'layers' must be a non-empty list.")
        self._layers = []
        for layer in layers:
            if isinstance(layer, tf.layers.Layer):
                self._layers.append(layer)
            else:
                self._layers.append(get_layer(hparams=layer))

        # Keep tracks of whether trainable variables have been created
        self._vars_built = False

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        for layer in self._layers:
            output_shape = layer.compute_output_shape(input_shape)
            input_shape = output_shape
        return output_shape

    def _collect_weights(self):
        """Collects (non-)trainable weights of each of the layers.
        """
        for layer in self._layers:
            if self.trainable:
                add_variable(
                    layer._trainable_weights, self._trainable_weights)
            else:
                add_variable(
                    layer._trainable_weights, self._non_trainable_weights)
            add_variable(
                layer._non_trainable_weights, self._non_trainable_weights)

    @property
    def trainable_weights(self):
        return self._trainable_weights

    @property
    def non_trainable_weights(self):
        return self._non_trainable_weights

    def call(self, inputs, mode=None): # pylint: disable=arguments-differ
        training = is_train_mode(mode)

        outputs = inputs
        for layer in self._layers:
            if isinstance(layer, tf.layers.Dropout) or \
                    isinstance(layer, tf.layers.BatchNormalization):
                outputs = layer(outputs, training=training)
            else:
                outputs = layer(inputs)
            inputs = outputs

        if not self.built or not self._vars_built:
            self._collect_weights()
            self._vars_built = True

        return outputs

    @property
    def layers(self):
        """The list of layers connected sequentially.
        """
        return self._layers


def _common_default_conv_dense_kwargs():
    """Returns the default keyword argument values that are common to
    convolution layers.
    """
    return {
        "activation": None,
        "use_bias": True,
        "kernel_initializer": {
            "type": "glorot_uniform_initializer",
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
        "bias_constraint": None,
        "trainable": True,
        "name": None
    }

def default_conv1d_kwargs():
    """Returns the default keyword argument values of the constructor
    of 1D-convolution layer class
    :tf_main:`tf.layers.Conv1D <layers/Conv1D>`.

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
                "type": "glorot_uniform_initializer",
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
            "trainable": True,
            "name": None
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
    """TODO
    """
    return {}
def default_conv3d_kwargs():
    """TODO
    """
    return {}
def default_conv2d_transpose_kwargs():
    """TODO
    """
    return {}
def default_conv3d_transpose_kwargs():
    """TODO
    """
    return {}

def default_dense_kwargs():
    """Returns the default keyword argument values of the constructor
    of the dense layer class :tf_main:`tf.layers.Dense <layers/Dense>`.

    .. code-block:: python

        {
            "units": 256,
            "activation": "identity",
            "use_bias": True,
            "kernel_initializer": {
                "type": "glorot_uniform_initializer",
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
            "trainable": True,
            "name": None
        }
    """
    kwargs = _common_default_conv_dense_kwargs()
    kwargs.update({
        "units": 256
    })
    return kwargs

def default_dropout_kwargs():
    """TODO
    """
    return {}
    #raise NotImplementedError
def default_flatten_kwargs():
    """TODO
    """
    return {}
def default_max_pooling1d_kwargs():
    """TODO
    """
    return {}
    #raise NotImplementedError
def default_max_pooling2d_kwargs():
    """TODO
    """
    return {}
    #raise NotImplementedError
def default_max_pooling3d_kwargs():
    """TODO
    """
    return {}
    #raise NotImplementedError
def default_separable_conv2d_kwargs():
    """TODO
    """
    return {}
    #raise NotImplementedError
def default_batch_normalization_kwargs():
    """TODO
    """
    return {}
    #raise NotImplementedError
def default_average_pooling1d_kwargs():
    """TODO
    """
    return {}
    #raise NotImplementedError
def default_average_pooling2d_kwargs():
    """TODO
    """
    return {}
    #raise NotImplementedError
def default_average_pooling3d_kwargs():
    """TODO
    """
    return {}
    #raise NotImplementedError
_layer_class_to_default_kwargs_map = {
    tf.layers.Conv1D: default_conv1d_kwargs(),
    tf.layers.Conv2D: default_conv2d_kwargs(),
    tf.layers.Conv3D: default_conv3d_kwargs(),
    tf.layers.Conv2DTranspose: default_conv2d_transpose_kwargs(),
    tf.layers.Conv3DTranspose: default_conv3d_transpose_kwargs(),
    tf.layers.Dense: default_dense_kwargs(),
    tf.layers.Dropout: default_dropout_kwargs(),
    tf.layers.Flatten: default_flatten_kwargs(),
    tf.layers.MaxPooling1D: default_max_pooling1d_kwargs(),
    tf.layers.MaxPooling2D: default_max_pooling2d_kwargs(),
    tf.layers.MaxPooling3D: default_max_pooling3d_kwargs(),
    tf.layers.SeparableConv2D: default_separable_conv2d_kwargs(),
    tf.layers.BatchNormalization: default_batch_normalization_kwargs(),
    tf.layers.AveragePooling1D: default_average_pooling1d_kwargs(),
    tf.layers.AveragePooling2D: default_average_pooling2d_kwargs(),
    tf.layers.AveragePooling3D: default_average_pooling3d_kwargs(),
}

def layer_normalize(inputs,
                    scope=None,
                    **kwargs):
    """Applies layer normalization. Normalizes over the last dimension.

    Args:
        inputs: A tensor with 2 or more dimensions, where the first
            dimension must be `batch_size`.
        scope (optional): variable scope.

    Returns:
        A tensor with the same shape and data dtype as `inputs`.
    """
    return tf.contrib.layers.layer_norm(
        inputs=inputs, begin_norm_axis=-1, begin_params_axis=-1, scope=scope,
        **kwargs
    )


def gelu(input_tensor):
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415

    Args:
      input_tensor: float Tensor to perform activation.

    Returns:
      `input_tensor` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
    return input_tensor * cdf
