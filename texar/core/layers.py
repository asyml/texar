"""
Various neural network layers
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math
import copy

import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from texar import context
from texar.hyperparams import HParams
from texar.utils import utils
import numpy as np
# pylint: disable=not-context-manager, redefined-variable-type, invalid-name
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
    "sinusoid_positional_encoding",
    "multihead_attention",
    "layer_normalize"
]

def default_rnn_cell_hparams():
    """Returns default hyperparameters of an RNN cell.

    Returns:
        A dictionary with the following structure and values:

        .. code-block:: python

            {
                # Name or full path of the cell class. E.g., the classname
                # of built-in cells in `tensorflow.contrib.rnn`, or the
                # classname of user-defined cells in `texar.custom`, or a
                # full path like "my_module.MyCell".
                "type": "BasicLSTMCell",

                # A dictionary of arguments for constructor of the cell
                # class. An RNN cell is created by calling the cell class
                # named in `type` passing the arguments specified in
                # `kwargs` as `cell_class(**kwargs)`
                "kwargs": {
                    "num_units": 256
                }

                # Number of cell layers
                "num_layers": 1

                # Dropout applied to the cell in each layer. See
                # `tensorflow.contrib.rnn.DropoutWrapper` for each of the
                # hyperparameters. If all keep probablities are 1.0, no dropout
                # is applied.
                "dropout": {
                    "input_keep_prob": 1.0,
                    "output_keep_prob": 1.0,
                    "state_keep_prob": 1.0,

                    # If True, the same dropout mask is applied at every step,
                    # and the list of input size of each layer is required
                    # (in "input_size"). The input size of a layer is the size
                    # of the last dimension of its input tensor. E.g., the
                    # input size of the first layer is usually the dimension of
                    # word embeddings, while the input size of followup layers
                    # are usually the num_units of the cells.
                    "variational_recurrent": False,
                    "input_size": []
                },

                # If True, apply residual connection on the inputs and
                # outputs of cell in each layer except the first layer.
                "residual": False,

                # If True, apply highway connection on the inputs and
                # outputs of cell in each layer except the first layer.
                "highway": False,
            }
    """
    return {
        "type": "BasicLSTMCell",
        "kwargs": {
            "num_units": 256,
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


def get_rnn_cell(hparams=None, mode=None):
    """Creates an RNN cell.

    See :meth:`~texar.core.layers.default_rnn_cell_hparams` for all
    hyperparameters and default values.

    Args:
        hparams (dict or HParams, optional): Cell hyperparameters. Missing
            hyperparameters are set to default values. If
            :attr:`hparams["type"]` is a cell instance (rather
            than the name or path to the cell class), then
            :attr:`hparams["num_layers"]` must be 1.
        mode (optional): A Tensor taking value in
            :tf_main:`tf.estimator.ModeKeys <estimator/ModeKeys>`, including
            `TRAIN`, `EVAL`, and `PREDICT`. If `None`, dropout will be
            controlled by :func:`texar.context.global_mode`.

    Returns:
        An instance of :tf_main:`RNNCell <contrib/rnn/RNNCell>`.

    Raises:
        ValueError: If :attr:`hparams["num_layers"]` > 1 and
            :attr:`hparams["type"]` is not of type string.
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
        if utils.is_str(cell_type):
            cell_modules = ['tensorflow.contrib.rnn', 'texar.custom']
            cell = utils.get_instance(cell_type, cell_kwargs, cell_modules)
        else:
            if num_layers > 1:
                raise ValueError(
                    "If `hparams['num_layers']`>1, then "
                    "`hparams['type']` must be a string name or path "
                    "to the class.")
            cell = cell_type
        if not isinstance(cell, rnn.RNNCell):
            raise ValueError("cell must be an instance of RNNCell.")

        # Optionally add dropout
        if d_hp["input_keep_prob"] < 1.0 or \
                d_hp["output_keep_prob"] < 1.0 or \
                d_hp["state_keep_prob"] < 1.0:
            vr_kwargs = {}
            if d_hp["variational_recurrent"]:
                vr_kwargs = {"variational_recurrent": True,
                             "input_size": d_hp["input_size"][layer_i],
                             "dtype": tf.float32}
            input_keep_prob = utils.switch_dropout(d_hp["input_keep_prob"],
                                                   mode)
            output_keep_prob = utils.switch_dropout(d_hp["output_keep_prob"],
                                                    mode)
            state_keep_prob = utils.switch_dropout(d_hp["state_keep_prob"],
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
        cell: an instance of :class:`tensorflow.contrib.rnn.RNNCell`.

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

    See :func:`~texar.core.layers.default_regularizer_hparams` for all
    hyperparameters and default values.

    Args:
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparameters are set to default values.

    Returns:
        A :tf_main:`Regularizer <keras/regularizers/Regularizer>` instance.
        `None` if :attr:`hparams` is `None` or takes the default
        hyperparameter value.

    Raises:
        ValueError: The resulting regularizer is not an instance of
            :tf_main:`Regularizer <keras/regularizers/Regularizer>`.
    """
    if hparams is None:
        return None

    if isinstance(hparams, dict):
        hparams = HParams(hparams, default_regularizer_hparams())
    if utils.is_str(hparams.type):
        rgl = utils.get_instance(
            hparams.type, hparams.kwargs.todict(),
            ["tensorflow.keras.regularizers", "texar.custom"])
    else:
        rgl = hparams.type
    if not isinstance(rgl, tf.keras.regularizers.Regularizer):
        raise ValueError("The regularizer must be an instance of "
                         "tf.keras.regularizers.Regularizer.")
    if isinstance(rgl, tf.keras.regularizers.L1L2) and \
            rgl.l1 == 0. and rgl.l2 == 0.:
        return None
    return rgl

def get_initializer(hparams=None):
    """Returns an initializer instance.

    Args:
        hparams (dict or HParams, optional): Hyperparameters.

    Returns:
        An initializer instance. `None` if :attr:`hparams` is `None`.
    """
    if hparams is None:
        return None

    if utils.is_str(hparams["type"]):
        #print('hparams:{}'.format(hparams))
        kwargs = hparams["kwargs"]
        if isinstance(kwargs, HParams):
            kwargs = kwargs.todict()
        modules = ["tensorflow.initializers", "tensorflow.keras.initializers",
                   "tensorflow", "texar.custom"]
        try:
            initializer = utils.get_instance(hparams["type"], kwargs, modules)
        except TypeError:
            modules += ['tensorflow.contrib.layers']
            initializer_fn = utils.get_function(hparams["type"], modules)
            initializer = initializer_fn(**kwargs)
    else:
        initializer = hparams["type"]
    return initializer

def get_activation_fn(fn_name="identity"):
    """Returns an activation function based on its name or full path.

    Args:
        fn_name (str or callable): The name or full path to the activation
            function.
            The function can be:

            - Built-in function defined in :mod:`tf` or \
              :mod:`tf.nn`, e.g., :tf_main:`identity <identity>`.
            - User-defined activation functions in `texar.custom`.
            - External activation functions. Must provide the full path, \
              e.g., "my_module.my_activation_fn".

            If a callable is provided, then it is returned directly.

    Returns:
        The activation function. `None` if :attr:`fn_name` is `None`.
    """
    if fn_name is None:
        return None

    if utils.is_callable(fn_name):
        return fn_name

    fn_modules = ['tensorflow', 'tensorflow.nn', 'texar.custom']
    activation_fn = utils.get_function(fn_name, fn_modules)
    return activation_fn

def get_constraint_fn(fn_name="NonNeg"):
    """Returns a constraint function based on its name or full path.

    Args:
        fn_name (str or callable): The name or full path to the
            constraint function.
            The function can be:

            - Built-in constraint functions defined in \
            :tf_main:`tf.keras.constraints <keras/constraints>` \
            (e.g., :tf_main:`NonNeg <keras/constraints/NonNeg>`) \
            or :mod:`tf` or :mod:`tf.nn` (e.g., activation functions).
            - User-defined function in :mod:`texar.custom`. The function \
            must follow the signature `w' = constraint_fn(w)`.
            - Externally defined function. Must provide the full path, \
            e.g., :attr:`"my_module.my_constraint_fn"`.

            If a callable is provided, then it is returned directly.

    Returns:
        The constraint function. `None` if :attr:`fn_name` is `None`.
    """
    if fn_name is None:
        return None

    if utils.is_callable(fn_name):
        return fn_name

    fn_modules = ['tensorflow.keras.constraints', 'tensorflow',
                  'tensorflow.nn', 'texar.custom']
    constraint_fn = utils.get_function(fn_name, fn_modules)
    return constraint_fn


#TODO: allow flat `type` and `kwargs` arguments.
def get_layer(hparams):
    """Makes a layer instance.

    The layer must be an instance of :tf_main:`Layer <layers/Layer>`.

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

            "type" : str or layer instance
                Name, full path, or instance of the layer class. The
                class can be

                - Built-in layer defined in \
                  :tf_main:`tf.layers <layers>` (e.g., \
                  :tf_main:`tf.layers.Conv2D <layers/Conv2D>`), or \
                  :mod:`tx.core <texar.core>` (e.g., \
                  :class:`tx.core.MergeLayer <texar.core.MergeLayer>`)
                - User-defined layer class in :mod:`tx.custom <texar.custom>`.\
                  The class must inherit :tf_main:`Layer <layers/Layer>`.
                - External layer. If str, must provide the full path, \
                  e.g., :attr:`"my_module.MyInitializer"`.

            "kwargs" : dict
                A dictionary of arguments for constructor of the
                layer class. Ignored if :attr:`"type"` is a layer instance.

                - Arguments named "activation" can be a callable, \
                or a `str` of \
                the name or full path to the activation function. \
                - Arguments named "*_regularizer" and "*_initializer" \
                can be a class instance, or a `dict` of \
                hyperparameters of \
                respective regularizers and initializers.
                - Arguments named "*_constraint" can be a callable, or a `str` \
                of the name or full path to the constraint function. \

    Returns:
        A layer instance. If :attr:`hparams["type"]` is already a layer
        instance, returns it directly.

    Raises:
        ValueError: If :attr:`hparams` is `None`.
        ValueError: If the resulting layer is not an instance of
            :tf_main:`Layer <layers/Layer>`.
    """
    if hparams is None:
        raise ValueError("`hparams` must not be `None`.")

    layer_type = hparams["type"]
    if not utils.is_str(layer_type):
        layer = layer_type
    else:
        layer_modules = ["tensorflow.layers", "texar.core", "texar.costum"]
        layer_class = utils.get_class(layer_type, layer_modules)
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
    """Pooling layer for abirary pooling functions for 1D inputs.

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
    """Max Pooling layer for 1D inputs. The same as
    :tf_main:`MaxPooling1D <layers/MaxPooling1D>` except that the pooling
    dimension is entirely reduced (i.e., `pool_size=length`).


    """
    def __init__(self, data_format='channels_last', name=None, **kwargs):
        super(MaxReducePooling1D, self).__init__(
            tf.reduce_max, data_format=data_format, name=name, **kwargs)

class AverageReducePooling1D(_ReducePooling1D):
    """Average Pooling layer for 1D inputs. The same as
    :tf_main:`AveragePooling1D <layers/AveragePooling1D>` except that the
    pooling dimension is entirely reduced (i.e., `pool_size=length`).


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
    """Creates pooling layer hparams dict usable for :func:`get_layer`.

    If the :attr:`hparams` sets `'pool_size'` to `None`, the layer will be
    changed to the respective reduce-pooling layer.
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
    """A layer that consists of multiple layers in parallel. Input is fed to
    each of the parallel layers, and the outputs are merged with a
    specified mode.

    Args:
        layers (list, optional): A list of :tf_main:`tf.layers.Layer
            <layers/layer>` instances, or a list of hyperparameter dicts
            each of which specifying type and kwargs of each layer (see
            the :attr:`hparams` argument of :func:`get_layer`). If `None`,
            inputs to the merge-layer directly merged.
        mode (str): Mode of the merge op. This can be:

            - :attr:`'concat'`: Concatenates layer outputs along one axis. \
              Tensors must have the same shape except for the dimension \
              specified in axis, which can have different sizes.
            - :attr:`'elemwise_sum'`: Outputs element-wise sum.
            - :attr:`'elemwise_mul'`: Outputs element-wise product.
            - :attr:`'sum'`: Computes the sum of layer outputs along the \
              dimension given in :attr:`axis`. E.g., given `axis=1`, \
              two tensors of shape `[a, b]` and `[a, c]` respectively \
              will result in a merged tensor of shape `[a]`.
            - :attr:`'mean'`: Computes the mean of layer outputs along the \
              dimension given in :attr:`axis`.
            - :attr:`'prod'`: Computes the product of layer outputs along the \
              dimension given in :attr:`axis`.
            - :attr:`'max'`: Computes the maximum of layer outputs along the \
              dimension given in :attr:`axis`.
            - :attr:`'min'`: Computes the minimum of layer outputs along the \
              dimension given in :attr:`axis`.
            - :attr:`'and'`: Computes the `logical and` of layer outputs along \
              the dimension given in :attr:`axis`.
            - :attr:`'or'`: Computes the `logical or` of layer outputs along \
              the dimension given in :attr:`axis`.
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

    def build(self, _):
        """Dumb method.
        """
        # Does not set :attr:`self.built` as this point.
        pass

    def _collect_weights(self):
        """Collects (non-)trainable weights of each of the parallel layers.
        """
        if self._layers is None:
            pass
        for layer in self._layers:
            if self.trainable:
                utils.add_variable(
                    layer._trainable_weights, self._trainable_weights)
            else:
                utils.add_variable(
                    layer._trainable_weights, self._non_trainable_weights)
            utils.add_variable(
                layer._non_trainable_weights, self._non_trainable_weights)

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

        if not self.built:
            self._collect_weights()

        return outputs

    @property
    def layers(self):
        """The list of parallel layers.
        """
        return self._layers


class SequentialLayer(tf.layers.Layer):
    """A layer that consists of multiple layers connected sequentially.

    Args:
        layers (list): A list of :tf_main:`tf.layers.Layer
            <layers/layer>` instances, or a list of hyperparameter dicts
            each of which specifying type and kwargs of each layer (see
            the :attr:`hparams` argument of :func:`get_layer`). The layers are
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

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        for layer in self._layers:
            output_shape = layer.compute_output_shape(input_shape)
            input_shape = output_shape
        return output_shape

    def build(self, _):
        """Dumb method.
        """
        # Does not set :attr:`self.built` as this point.
        pass

    def _collect_weights(self):
        """Collects (non-)trainable weights of each of the layers.
        """
        for layer in self._layers:
            if self.trainable:
                utils.add_variable(
                    layer._trainable_weights, self._trainable_weights)
            else:
                utils.add_variable(
                    layer._trainable_weights, self._non_trainable_weights)
            utils.add_variable(
                layer._non_trainable_weights, self._non_trainable_weights)

    def call(self, inputs):
        outputs = inputs
        for layer in self._layers:
            outputs = layer(inputs)
            inputs = outputs

        if not self.built:
            self._collect_weights()

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
        "kernel_regularizer": default_regularizer_hparams(),
        "bias_regularizer": default_regularizer_hparams(),
        "activity_regularizer": default_regularizer_hparams(),
        "kernel_constraint": None,
        "bias_constraint": None,
        "trainable": True,
        "name": None
    }

#TODO(zhiting): fix the docstring
def default_conv1d_kwargs():
    """Returns the default keyword argument values of 1D convolution layer
    defined in :tf_main:`tf.layers.Conv1D <layers/Conv1D>`.

    Returns:
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

        Here:

        "filters" : int
            The number of filters in the convolution.

            The default value is `100`.

        "kernel_size" : int
            The length of 1D convolution window.

            The default value is `3`.

        "strides" : int
            The stride length of the convolution.

            The default value is `1`.

        "padding" : str
            One of `"valid"` or `"same"` (case-insensitive).

            The default value is `"valid"`.

        "data_format" : str
            The ordering of the dimensions in the inputs. One of
            `"channels_last"` or `"channels_first"`.
            `"channels_last"` corresponds to inputs with shape
            `(batch, length, channels)`; `"channels_first"` corresponds to
            inputs with shape `(batch, channels, length)`.

            The default value is `"channels_last"`.

        "dilation_rate" : int
            The dilation rate to use for dilated convolution.

            The default value is `1`.

        "activation" : str
            The name or full path to the activation function applied to the
            outputs of the layer.

            The default value is "identity", which corr. to
            :tf_main:`tf.identity <identity>`.

        "kernel_initializer" : dict
            Hyperparameters of the initializer for the filters, including
            :attr:`"type"` (str or object) and :attr:`"kwargs"` (dict).

            The default corr. to :tf_main:`tf.glorot_uniform_initializer
            <glorot_uniform_initializer>`.

        "bias_initializer" : dict
            Hyperparameters of the initializer for the bias, including
            :attr:`"type"` (str or object) and :attr:`"kwargs"` (dict).

            The default corr. to
            :tf_main:`tf.zeros_initializer <zeros_initializer>`.

        "kernel_regularizer" : dict
            Optional hyperparameters of the regularizer for the convolution
            filters, including :attr:`"type"` (str or object) and
            :attr:`"kwargs"` (dict).

            The default value disables regularization.

        "bias_regularizer" : dict
            Optional hyperparameters of the regularizer for the bias,
            including :attr:`"type"` (str or object) and
            :attr:`"kwargs"` (dict).

            The default value disables regularization.

        "activity_regularizer" : dict
            Optional hyperparameters of the regularizer for the layer output,
            including :attr:`"type"` (str or object) and
            :attr:`"kwargs"` (dict).

            The default value disables regularization.

        "kernel_constraint" : str
            Optional name or full path to projection function to be applied to
            the kernel after being updated by an `Optimizer`. Used to
            implement norm constraints
            or value constraints for layer weights. The function must take
            as input the unprojected variable and return the projected variable
            with the same shape. Constraints are not safe to use when doing
            asynchronous distributed training.

            The function can be:

            - Built-in constraint functions defined in \
            :tf_main:`tf.keras.constraints <keras/constraints>` \
            (e.g., :tf_main:`NonNeg <keras/constraints/NonNeg>`) \
            or :mod:`tf` or :mod:`tf.nn` (e.g., activation functions).
            - User-defined function in :mod:`texar.custom`. The function \
            must follow the signature `w' = constraint_fn(w)`.
            - Externally defined function. Must provide the full path, \
            e.g., :attr:`"my_module.my_function"`.

            The default value is `None`.

        "bias_constraint" : str
            Optional name or full path to projection function to be applied to
            the bias after being updated by an `Optimizer`.

            The default value is `None`.
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
    """Returns the default keyword argument values of dense layer
    defined in :tf_main:`tf.layers.Dense <layers/Dense>`.
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



def sinusoid_positional_encoding(inputs,
                                 reuse=None,
                                 min_timescale=1.0,
                                 max_timescale=1.0e4,
                                 variable_scope='sinuoid_positional_embedding'):
    """obtain a positional encoding of inputs
    Args:
        inputs: [Tensor] A Tensor of shape `[batch_size, max_time, hidden_dim]`
        variable_scope: [String], Optional scope for 'variable_scope'
    """
    length = tf.shape(inputs)[1]
    channels = tf.shape(inputs)[2]
    with tf.variable_scope(variable_scope, reuse=reuse):
        position = tf.to_float(tf.range(length))
        num_timescales = channels // 2
        log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (tf.to_float(num_timescales) - 1))
        inv_timescales = min_timescale * tf.exp(
            tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
        scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
        signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
        signal = tf.reshape(signal, [1, length, channels])
        return signal

def multihead_attention(queries,
                        bias=None,
                        memory=None,
                        causality=False,
                        num_heads=8,
                        num_units=None,
                        dropout_rate=0,
                        cache=None,
                        scope='multihead_attention'):
    '''Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [batch, length_query, depth_query].
      keys: A 3d tensor with shape of [batch, length_key, depth_key].
      num_units: A scalar indicating the attention size, equals to depth_query if not given.
      dropout_rate: A floating point number.
      causality: Boolean. If true, units that reference the future are masked.
      num_heads: An int. Number of heads with calculating attention.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    Returns
      A 3d tensor with shape of (batch, length_query, num_units)
    '''
    with tf.variable_scope(scope):
        if num_units is None:
            num_units = queries.get_shape().as_list()[-1]
        if num_units % num_heads != 0:
            raise ValueError("Value depth (%d) must be divisible by the number"
                             "of attention heads (%d)." % (\
                            num_units, num_heads))
        if memory is None:
            #'self attention'
            Q = tf.layers.dense(queries, num_units, use_bias=False, name='q')
            K = tf.layers.dense(queries, num_units, use_bias=False, name='k')
            V = tf.layers.dense(queries, num_units, use_bias=False, name='v')
            if cache is not None:
                # 'decoder self attention'
                K = cache['self_keys'] = tf.concat([cache['self_keys'], K], axis=1)
                V = cache['self_values'] = tf.concat([cache['self_values'], V], axis=1)
                cache['self_keys'] = K
                cache['self_values'] = V
        else:
            # 'encoder decoder attention'
            Q = tf.layers.dense(queries, num_units, use_bias=False, name='q')
            if cache is not None:
                K, V = tf.cond(
                    tf.equal(tf.shape(cache["memory_keys"])[1], 0),
                    true_fn=lambda: [tf.layers.dense(memory, num_units, use_bias=False, name='k'),
                        tf.layers.dense(memory, num_units, use_bias=False, name='v')],
                    false_fn=lambda: [cache["memory_keys"], cache["memory_values"]])
            else:
                K, V = [tf.layers.dense(memory, num_units, use_bias=False, name='k'),
                        tf.layers.dense(memory, num_units, use_bias=False, name='v')]
        Q_ = split_heads(Q, num_heads)
        #[batch_size, num_heads, seq_length, memory_depth]
        K_ = split_heads(K, num_heads)
        V_ = split_heads(V, num_heads)
        key_depth_per_head = num_units // num_heads
        Q_ *= key_depth_per_head**-0.5

        logits = tf.matmul(Q_, K_, transpose_b=True)
        if bias is not None:
            logits += bias
        weights = tf.nn.softmax(logits, name="attention_weights")
        weights = tf.layers.dropout(
            weights, rate=dropout_rate, training=context.global_mode_train())

        outputs = tf.matmul(weights, V_)

        outputs = combine_heads(outputs)
        outputs = tf.layers.dense(outputs, num_units, use_bias=False, name='output_transform')
        #(batch_size, length_query, attention_depth)
    return outputs

def layer_normalize(inputs,
              #epsilon=1e-6, it seems in t2t, it's 1e-6
              epsilon=1e-8,
              scope='ln',
              reuse=None):
    '''Applies layer normalization. averaging over the last dimension
    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        filters = inputs.get_shape()[-1]
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        scale = tf.get_variable('layer_norm_scale', [filters], initializer=tf.ones_initializer())
        bias = tf.get_variable('layer_norm_bias', [filters], initializer=tf.zeros_initializer())
        norm_x = (inputs - mean) * tf.rsqrt(variance + epsilon)
        outputs = norm_x * scale + bias
    return outputs


def split_heads(x, num_heads):
    """Split channels (dimension 2) into multiple heads, becomes dimension 1).
        must ensure x.shape[-1] can be deviced by num_heads.any
    """
    depth = x.get_shape()[-1]
    splitted_x = tf.reshape(x, [tf.shape(x)[0], tf.shape(x)[1], \
        num_heads, depth // num_heads])
    return tf.transpose(splitted_x, [0, 2, 1, 3])
def combine_heads(x):
    """
    input: [batch, num_heads, seq_len, dim]
    output:[batch, seq_len, num_heads*dim]
    """
    t = tf.transpose(x, [0, 2, 1, 3]) #[batch, seq_len, num_heads, dim]
    num_heads, dim = t.get_shape()[-2:]
    return tf.reshape(t, [tf.shape(t)[0], tf.shape(t)[1], num_heads*dim])

def shape_list(x):
  """Return list of dims, statically where possible."""
  x = tf.convert_to_tensor(x)

  # If unknown rank, return dynamic shape
  if x.get_shape().dims is None:
    return tf.shape(x)

  static = x.get_shape().as_list()
  shape = tf.shape(x)

  ret = []
  for i in range(len(static)):
    dim = static[i]
    if dim is None:
      dim = shape[i]
    ret.append(dim)
  return ret

def ones_matrix_band_part(rows, cols, num_lower, num_upper, out_shape=None):
  """Matrix band part of ones."""
  if all([isinstance(el, int) for el in [rows, cols, num_lower, num_upper]]):
    # Needed info is constant, so we construct in numpy
    if num_lower < 0:
      num_lower = rows - 1
    if num_upper < 0:
      num_upper = cols - 1
    lower_mask = np.tri(cols, rows, num_lower).T
    upper_mask = np.tri(rows, cols, num_upper)
    band = np.ones((rows, cols)) * lower_mask * upper_mask
    if out_shape:
      band = band.reshape(out_shape)
    band = tf.constant(band, tf.float32)
  else:
    band = tf.matrix_band_part(tf.ones([rows, cols]),
                               tf.cast(num_lower, tf.int64),
                               tf.cast(num_upper, tf.int64))
    if out_shape:
      band = tf.reshape(band, out_shape)

  return band
