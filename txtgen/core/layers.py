"""
Various neural network layers
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import numpy as np

from txtgen import context
from txtgen.hyperparams import HParams
from txtgen.core import utils

# pylint: disable=not-context-manager, redefined-variable-type, invalid-name
# pylint: disable=too-many-branches

def default_rnn_cell_hparams():
    """Returns default hyperparameters of an RNN cell.

    Returns:
        A dictionary with the following structure and values:

        .. code-block:: python

            {
                "cell": {
                    # Name or full path of the cell class. E.g., the classname
                    # of built-in cells in `tensorflow.contrib.rnn`, or the
                    # classname of user-defined cells in `txtgen.custom`, or a
                    # full path like "my_module.MyCell".
                    "type": "BasicLSTMCell",

                    # A dictionary of arguments for constructor of the cell
                    # class. An RNN cell is created by calling the cell class
                    # named in `type` passing the arguments specified in
                    # `kwargs` as `cell_class(**kwargs)`
                    "kwargs": {
                        "num_units": 64
                    }
                },

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
        "cell": {
            "type": "BasicLSTMCell",
            "kwargs": {
                "num_units": 64
            }
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


def get_rnn_cell(hparams=None):
    """Creates an RNN cell.

    See :meth:`~txtgen.core.layers.default_rnn_cell_hparams` for all
    hyperparameters and default values.

    Args:
        hparams (dict or HParams, optional): Cell hyperparameters. Missing
            hyperparameters are set to default values. If
            :attr:`hparams["cell"]["type"]` is a cell instance (rather
            than the name or path to the cell class), then
            :attr:`hparams["num_layers"]` must be 1.

    Returns:
        An instance of :tf_main:`RNNCell <contrib/rnn/RNNCell>`.

    Raises:
        ValueError: If :attr:`hparams["num_layers"]` > 1 and
            :attr:`hparams["cell"]["type"]` is not of type string.
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
    cell_kwargs = hparams["cell"]["kwargs"].todict()
    num_layers = hparams["num_layers"]
    for layer_i in range(num_layers):
        # Create the basic cell
        cell_type = hparams["cell"]["type"]
        if utils.is_str_or_unicode(cell_type):
            cell_modules = ['tensorflow.contrib.rnn', 'txtgen.custom']
            cell = utils.get_instance(cell_type, cell_kwargs, cell_modules)
        else:
            if num_layers > 1:
                raise ValueError(
                    "If `hparams['num_layers']`>1, then "
                    "`hparams['cell']['type']` must be a string name or path "
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
            cell = rnn.DropoutWrapper(
                cell=cell,
                input_keep_prob=utils.switch_dropout(d_hp["input_keep_prob"]),
                output_keep_prob=utils.switch_dropout(d_hp["output_keep_prob"]),
                state_keep_prob=utils.switch_dropout(d_hp["state_keep_prob"]),
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

def _default_regularizer_hparams():
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

    See :meth:`~txtgen.core.layers.default_regularizer_hparams` for all
    hyperparameters and default values.

    Args:
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparameters are set to default values.

    Returns:
        A :tf_main:`Regularizer <keras/regularizers/Regularizer>` instance.
        `None` if :attr:`hparams` takes the default value.

    Raises:
        ValueError: The resulting regularizer is not an instance of
            :tf_main:`Regularizer <keras/regularizers/Regularizer>`.
    """
    if hparams is None:
        return None
    if isinstance(hparams, dict):
        hparams = HParams(hparams, _default_regularizer_hparams())
    if utils.is_str_or_unicode(hparams.type):
        rgl = utils.get_instance(
            hparams.type, hparams.kwargs.todict(),
            ["tensorflow.keras.regularizers", "txtgen.custom"])
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
    if utils.is_str_or_unicode(hparams["type"]):
        kwargs = hparams["kwargs"]
        if isinstance(kwargs, HParams):
            kwargs = kwargs.todict()
        initializer = utils.get_instance(
            hparams["type"], kwargs,
            ["tensorflow.initializers", "tensorflow", "txtgen.custom"])
    else:
        initializer = hparams["type"]
    return initializer

def get_activation_fn(fn_name="identity"):
    """Returns an activation function based on its name or full path.

    Args:
        fn_name (str): The name or full path to the activation function.
            The function can be:

            - Built-in function defined in :mod:`tf` or \
              :mod:`tf.nn`, e.g., :tf_main:`identity <identity>`.
            - User-defined activation functions in `txtgen.custom`.
            - External activation functions. Must provide the full path, \
              e.g., "my_module.my_activation_fn".

            The default value is "identity".

    Returns:
        The activation function.
    """
    fn_modules = ['tensorflow', 'tensorflow.nn', 'txtgen.custom']
    activation_fn = utils.get_function(fn_name, fn_modules)
    return activation_fn

def default_embedding_hparams():
    """Returns default hyperparameters of token embedding used in encoders,
    decoders, and other modules.

    Returns:
        A dictionary with the following structure and values.

        .. code-block:: python

            {
                "name": "embedding",
                "dim": 100,
                "initializer": {
                    "type": "random_uniform_initializer",
                    "kwargs": {
                        "minval": -0.1,
                        "maxval": 0.1,
                        "seed": None
                    }
                },
                "regularizer": {
                    "type": "L1L2",
                    "kwargs": {
                        "l1": 0.,
                        "l2": 0.
                    }
                }
                "trainable": True,
            }

        Here:

        "name" : str
            Name of the embedding variable.

        "dim" : int
            Embedding dimension.

        "initializer" : dict
            Hyperparameters of the initializer for the embedding values,
            including:

            "type" : str or initializer instance
                Name, full path, or instance of the initializer class. The
                class can be

                - Built-in initializer defined in
                  :tf_main:`tf.initializers <initializers>`, e.g.,
                  :tf_main:`random_uniform <random_uniform_initializer>` (a.k.a
                  tf.random_uniform_initializer) or in
                  :mod:`tf`, e.g., :tf_main:`glorot_uniform_initializer
                  <glorot_uniform_initializer>`.
                - User-defined initializer in :mod:`txtgen.custom`.
                - External initializer. Must provide the full path, \
                  e.g., :attr:`"my_module.MyInitializer"`, or the instance.

            "kwargs" : dict
                A dictionary of arguments for constructor of the
                initializer class. An initializer is created by
                calling `initialzier_class(**kwargs)` where
                :attr:`initializer_class` is specified in :attr:`"type"`.
                Ignored if :attr:`"type"` is an initializer instance.

            The default value corresponds to the initializer
            :tf_main:`tf.random_uniform_initializer
            <random_uniform_initializer>`.

        "regularizer" : dict
            Hyperparameters of the regularizer for the embedding values. The
            regularizer must be an instance of
            the base :tf_main:`Regularizer <keras/regularizers/Regularizer>`
            class. The hyperparameters include:

            "type" : str or Regularizer instance
                Name, full path, or instance of the regularizer class. The
                class can be

                - Built-in regularizer defined in
                  :tf_main:`tf.keras.regularizers <keras/regularizers>`, e.g.,
                  :tf_main:`L1L2 <keras/regularizers/L1L2>`.
                - User-defined regularizer in :mod:`txtgen.custom`. The
                  regularizer class should inherit the base class
                  :tf_main:`Regularizer <keras/regularizers/Regularizer>`.
                - External regularizer. Must provide the full path, \
                  e.g., :attr:`"my_module.MyRegularizer"`, or the instance.

            "kwargs" : dict
                A dictionary of arguments for constructor of the
                regularizer class. A regularizer is created by
                calling `regularizer_class(**kwargs)` where
                :attr:`regularizer_class` is specified in :attr:`"type"`.
                Ignored if :attr:`"type"` is a Regularizer instance.

            The default value corresponds to
            :tf_main:`L1L2 <keras/regularizers/L1L2>` with `(l1=0, l2=0)`,
            which disables regularization.

        "trainable" : bool
            Whether the embedding is trainable.
    """
    return {
        "name": "embedding",
        "dim": 50,
        "initializer": {
            "type": "random_uniform_initializer",
            "kwargs": {
                "minval": -0.1,
                "maxval": 0.1,
                "seed": None
            }
        },
        "regularizer": _default_regularizer_hparams(),
        "trainable": True
    }


def get_embedding(hparams=None,
                  init_values=None,
                  vocab_size=None,
                  variable_scope=None):
    """Creates embedding variable if not exists.

    Args:
        hparams (dict or HParams, optional): Embedding hyperparameters. Missing
            hyperparameters are set to default values. See
            :func:`~txtgen.core.layers.default_embedding_hparams` for all
            hyperparameters and default values.

            If :attr:`init_values` is given, :attr:`hparams["initializer"]`,
            and :attr:`hparams["dim"]` are ignored.
        init_values (Tensor or numpy array, optional): Initial values of the
            embedding variable. If not given, embedding is initialized as
            specified in :attr:`hparams["initializer"]`.
        vocab_size (int, optional): The vocabulary size. Required if
            :attr:`init_values` is not provided.
        variable_scope (string or VariableScope, optional): Variable scope of
            the embedding variable.

    Returns:
        Variable: A 2D `Variable` of the same shape with :attr:`init_values`
        or of the shape :attr:`[vocab_size, hparams["dim"]]`.
    """
    with tf.variable_scope(variable_scope, "embedding"):
        if hparams is None or isinstance(hparams, dict):
            hparams = HParams(hparams, default_embedding_hparams())
        regularizer = get_regularizer(hparams["regularizer"])
        if init_values is None:
            initializer = get_initializer(hparams["initializer"])
            return tf.get_variable(name=hparams["name"],
                                   shape=[vocab_size, hparams["dim"]],
                                   initializer=initializer,
                                   regularizer=regularizer,
                                   trainable=hparams["trainable"])
        else:
            return tf.get_variable(name=hparams["name"],
                                   initializer=init_values,
                                   regularizer=regularizer,
                                   trainable=hparams["trainable"])

#TODO(zhiting): checkout
# https://github.com/Lasagne/Lasagne/blob/master/lasagne/layers/merge.py#L243
class MergeLayer(tf.layers.Layer):
    """A layer that consists of multiple layers in parallel. Input is feed to
    each of the sub-layers, and the outputs are merged with a specified mode.
    """
    #raise NotImplementedError
    pass

def _common_default_conv_kwargs():
    """Returns the default keyword argument values that are common to
    convolution layers.
    """
    return {
        "activation": "identity",
        "kernel_initializer": {
            "type": "glorot_uniform_initializer",
            "kwargs": {}
        },
        "bias_initializer": {
            "type": "zeros_initializer",
            "kwargs": {}
        },
        "kernel_regularizer": _default_regularizer_hparams(),
        "bias_regularizer": _default_regularizer_hparams(),
        "activity_regularizer": _default_regularizer_hparams()
    }

#TODO(zhiting): fix the docstring
def default_conv1d_kwargs():
    """Returns the default keyword argument values of 1D convolution layer(s)
    defined in :tf_main:`tf.layers.Conv1D <layers/Conv1D>`.

    Some of the keyword arguments allow extended values as detailed in the
    following.

    Returns:
        .. code-block:: python

            {
                "kernel_size": 3,
                "filters": 100,
                "strides": 1,
                "activation": "identity",
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
                }
                "bias_regularizer": {
                    # same as in "kernel_regularizer"
                    # ...
                },
                "activity_regularizer": {
                    # same as in "kernel_regularizer"
                    # ...
                }
            }

        Here:

        "kernel_size" : int or a list of int
            The length(s) of 1D convolution window(s). If a list, filters with
            different window lengths as specified in the list are created.

            The default value is `[3,4,5]`, which creates 3 sets of filters,
            each of which are with lengths 3, 4, and 5.

        "filters" : int or a list of int
            The number of filters in the convolution. If an int, equal number of
            filters with different window lengths are created. If a list,
            the list must be of the same length as the list in
            :attr:`"kernel_size"`, and each integer in the list is the number
            of filters with respective window length.

            The default value is `100`, which creates 100 filters for each
            filter set.

        "strides" : int or a list of int
            The stride length of the convolution. If an int, the stride length
            is shared across all filter sets. If a list, the list must be of
            the same length as the list in :attr:`"kernel_size"`.

            The default value is `1`.

        "dilation_rate" : int or a list of int
            The dilation rate to use for dilated convolution. If an int, the
            dilation rate is shared across all filter sets. If a list, the list
            must be of the same length as the list in :attr:`"kernel_size"`.

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
    """
    return {
        "kernel_size": 3,
        "filters": 100,
        "strides": 1,
        "dilation_rate": 1,
    }

def default_conv2d_kwargs():
    return {}
    #raise NotImplementedError
def default_conv3d_kwargs():
    return {}
    #raise NotImplementedError
def default_conv2d_transpose_kwargs():
    return {}
    #raise NotImplementedError
def default_conv3d_transpose_kwargs():
    return {}
    #raise NotImplementedError
def default_dense_kwargs():
    return {}
    #raise NotImplementedError
def default_dropout_kwargs():
    return {}
    #raise NotImplementedError
def default_flatten_kwargs():
    return {}
    #raise NotImplementedError
def default_max_pooling1d_kwargs():
    return {}
    #raise NotImplementedError
def default_max_pooling2d_kwargs():
    return {}
    #raise NotImplementedError
def default_max_pooling3d_kwargs():
    return {}
    #raise NotImplementedError
def default_separable_conv2d_kwargs():
    return {}
    #raise NotImplementedError
def default_batch_normalization_kwargs():
    return {}
    #raise NotImplementedError
def default_average_pooling1d_kwargs():
    return {}
    #raise NotImplementedError
def default_average_pooling2d_kwargs():
    return {}
    #raise NotImplementedError
def default_average_pooling3d_kwargs():
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

def get_layer(hparams):
    """Makes a layer instance.

    The layer must be an instance of :tf_main:`Layer <layers/Layer>`.

    Args:
        hparams (dict or HParams): Hyperparameters of the layer, with
            structure:

            .. code-block:: python

                {
                    "type": "layer_class",
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
                  :tf_main:`tf.layers <layers>`, e.g., \
                  :tf_main:`tf.layers.Conv2D <layers/Conv2D>`. \
                - User-defined layer class in :mod:`txtgen.custom`. The class \
                  must inherit :tf_main:`Layer <layers/Layer>`.
                - External layer. Must provide the full path, \
                  e.g., :attr:`"my_module.MyInitializer"`, or the instance.

            "kwargs" : dict
                A dictionary of arguments for constructor of the
                layer class. Ignored if :attr:`"type"` is a layer instance.

                - Arguments named "*_regularizer" and "*_initializer" \
                have values of type `dict` that specifiy hyperparameters of \
                respective regularizers and initializers. Regularizer and \
                initializer instances will be created accordingly and used for \
                making the layer.
                - Arguments named "activation" have values of type `str` that \
                specify the name or full path to the activation function. \
                The activation function will be used for making the layer.

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
    if not utils.is_str_or_unicode(layer_type):
        layer = layer_type
    else:
        layer_modules = ["tensorflow.layers", "txtgen.costum"]
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
            else:
                kwargs[k] = v

        layer = utils.get_instance(layer_type, kwargs, layer_modules)

    if not isinstance(layer, tf.layers.Layer):
        raise ValueError("layer must be an instance of `tf.layers.Layer`.")

    return layer


#TODO(zhiting): fix code style
def sinusoid_positional_encoding(inputs,
                                zero_pad=True,
                                scale=True,
                                reuse=None,
                                position_duration=10000,
                                max_time=None,
                                scope='sinuoid_positional_embedding'):
    """obtain a positional encoding of inputs
    Args:
        inputs: [Tensor] A Tensor of shape `[batch_size, max_time, hidden_dim]`
        max_time: [Int], max time steps
        hidden_dim: [Int], hidden size of embedding
        zero_pad: [Boolean], If True, all the values of the first row(id = 0) should be constant zero
        scale: [Boolean], If True, the output will be multiplied by sqrt num_units(check details from paper)
        scope: [String], Optional scope for 'variable_scope'
        position_duration: [Int], default=10000
    """
    print('inputs:shape{}'.format(inputs.get_shape())) #[3, None, 50]
    batch_size, _, hidden_dim = inputs.get_shape().as_list()
    with tf.variable_scope(scope, reuse=reuse):
        position_idx = tf.tile(tf.expand_dims(tf.range(max_time), 0), [batch_size, 1]) #batch_size * max_time
        position_enc = np.array([
            [pos /np.power(10000, 2.*i/hidden_dim) for i in range(hidden_dim)]
            for pos in range(max_time)])

        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])

        lookup_table = tf.convert_to_tensor(position_enc, dtype=tf.float32)
        print('lookup_table:{}'.format(lookup_table))
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, hidden_dim]),
                lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, position_idx)
        if scale:
            outputs = outputs * hidden_dim**0.5
        print('outputs:{}'.format(outputs.shape))
        return outputs

#TODO(zhiting): fix code style
def multihead_attention(queries,
                        keys,
                        num_heads=8,
                        dropout_rate=0,
                        causality=False,
                        scope = 'multihead_attention',
                        reuse= None):
    """perform multihead attention
    Args:
        queries: A 3d tensor with shape of [N, T_q, C_q].
        keys: A 3d tensor with shape of [N, T_k, C_k].
        num_units: A scalar. Attention size.
        dropout_rate: A floating point number.
        causality: Boolean. Should be true, units that reference the future are masked
        num_heads: An int. Number of heads.
        scope: Optional scope for `variable_scope`.
        reuse: Boolean, whether to reuse the weights of a previous layer by the same name.
    Returns
        A 3d tensor with shape of (N, T_q, C)
    """
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        num_units = queries.get_shape().as_list()[-1]

        # Linear projections
        print('keys:{}'.format(keys))
        print('queries:{}'.format(queries))
        print('num_units:{}'.format(num_units))
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu) # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*N, T_q, T_k)

        # According to the paper, there is a scale operation
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1))) # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs)*(-2**32+1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :]) # (T_q, T_k)
            tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense() # (T_q, T_k)  The upper triangle of the last two dimensions is ignored.
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]) # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks)*(-2**32+1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs) # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs) # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1))) # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) # (h*N, T_q, T_k)
        outputs *= query_masks # broadcasting. (N, T_q, C)

        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=context.is_train())

        # Weighted sum
        outputs = tf.matmul(outputs, V_) # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, C)

        # Residual connection
        outputs += queries

        # Normalize

    return outputs



def poswise_feedforward(attended_dec, scope="multihead_attention", reuse=None):
    '''Point-wise feed forward net.

    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    hidden_dim = attended_dec.shape.as_list()[-1]
    with tf.variable_scope(scope, reuse=reuse):
        outputs = tf.layers.conv1d(inputs = attended_dec,
                filters=hidden_dim*4,
                kernel_size=1,
                activation=tf.nn.relu,
                use_bias=True)
        outputs = tf.layers.conv1d(inputs = outputs,
                filters=hidden_dim,
                kernel_size=1,
                activation=None,
                use_bias=True)
        outputs += attended_dec #residual connection
    return outputs

def normalize(inputs,
              epsilon = 1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.

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
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta= tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        outputs = gamma * normalized + beta

    return outputs

