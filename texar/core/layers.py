"""
Various neural network layers
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import numpy as np

from texar import context
from texar.hyperparams import HParams
from texar.core import utils

# pylint: disable=not-context-manager, redefined-variable-type, invalid-name
# pylint: disable=too-many-branches, too-many-arguments, too-many-lines

def default_rnn_cell_hparams():
    """Returns default hyperparameters of an RNN cell.

    Returns:
        A dictionary with the following structure and values:

        .. code-block:: python

            {
                "cell": {
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

    See :meth:`~texar.core.layers.default_rnn_cell_hparams` for all
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
            cell_modules = ['tensorflow.contrib.rnn', 'texar.custom']
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

    See :meth:`~texar.core.layers.default_regularizer_hparams` for all
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
        hparams = HParams(hparams, _default_regularizer_hparams())
    if utils.is_str_or_unicode(hparams.type):
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

    if utils.is_str_or_unicode(hparams["type"]):
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
        fn_name (str): The name or full path to the activation function.
            The function can be:

            - Built-in function defined in :mod:`tf` or \
              :mod:`tf.nn`, e.g., :tf_main:`identity <identity>`.
            - User-defined activation functions in `texar.custom`.
            - External activation functions. Must provide the full path, \
              e.g., "my_module.my_activation_fn".

            The default value is "identity".

    Returns:
        The activation function. `None` if :attr:`fn_name` is `None`.
    """
    if fn_name is None:
        return None

    fn_modules = ['tensorflow', 'tensorflow.nn', 'texar.custom']
    activation_fn = utils.get_function(fn_name, fn_modules)
    return activation_fn

def get_constraint_fn(fn_name="NonNeg"):
    """Returns a constraint function based on its name or full path.

    Args:
        fn_name (str): The name or full path to the constraint function.
            The function can be:

            - Built-in constraint functions defined in \
            :tf_main:`tf.keras.constraints <keras/constraints>` \
            (e.g., :tf_main:`NonNeg <keras/constraints/NonNeg>`) \
            or :mod:`tf` or :mod:`tf.nn` (e.g., activation functions).
            - User-defined function in :mod:`texar.custom`. The function \
            must follow the signature `w' = constraint_fn(w)`.
            - Externally defined function. Must provide the full path, \
            e.g., :attr:`"my_module.my_constraint_fn"`.

    Returns:
        The constraint function. `None` if :attr:`fn_name` is `None`.
    """
    if fn_name is None:
        return None

    fn_modules = ['tensorflow.keras.constraints', 'tensorflow',
                  'tensorflow.nn', 'texar.custom']
    constraint_fn = utils.get_function(fn_name, fn_modules)
    return constraint_fn

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
                Name, full path, or instance of the initializer class; Or name
                or full path to a function that returns the initializer class.
                The class or function can be

                - Built-in initializer defined in \
                  :tf_main:`tf.initializers <initializers>`, e.g., \
                  :tf_main:`random_uniform <random_uniform_initializer>` \
                  (a.k.a :class:`tf.random_uniform_initializer`), or \
                  in :mod:`tf`, e.g., :tf_main:`glorot_uniform_initializer \
                  <glorot_uniform_initializer>`, or in \
                  :tf_main:`tf.keras.initializers <keras/initializers>`.
                - User-defined initializer in :mod:`texar.custom`.
                - External initializer. Must provide the full path, \
                  e.g., :attr:`"my_module.MyInitializer"`, or the instance.

            "kwargs" : dict
                A dictionary of arguments for constructor of the
                initializer class or for the function. An initializer is
                created by `initialzier = initializer_class_or_fn(**kwargs)`
                where :attr:`initializer_class_or_fn` is specified in
                :attr:`"type"`.
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
                - User-defined regularizer in :mod:`texar.custom`. The
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
                  variable_scope='Embedding'):
    """Creates embedding variable if not exists.

    Args:
        hparams (dict or HParams, optional): Embedding hyperparameters. Missing
            hyperparameters are set to default values. See
            :func:`~texar.core.layers.default_embedding_hparams` for all
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
    with tf.variable_scope(variable_scope):
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
                                   initializer=tf.to_float(init_values),
                                   regularizer=regularizer,
                                   trainable=hparams["trainable"])

#TODO(zhiting): checkout
# https://github.com/Lasagne/Lasagne/blob/master/lasagne/layers/merge.py#L243
class MergeLayer(tf.layers.Layer):
    """A layer that consists of multiple layers in parallel. Input is fed to
    each of the sub-layers, and the outputs are merged with a specified mode.

    Args:
        layers (list):
    """

    def __init__(self,
                 layers,
                 mode='concat',
                 axis=0,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(MergeLayer, self).__init__(
            trainable=trainable, name=name, **kwargs)
        self._layers = layers
        self._mode = mode
        self._axis = axis

    def build(self, input_shape):
        """Creates the variables of the layer.

        Overloads the base class :tf_main:`tf.layers.Layer <layers/Layer>`.
        """
        #input_shape = tensor_shape.TensorShape(input_shape)
        #if input_shape[-1].value is None:
        #  raise ValueError('The last dimension of the inputs to `Dense` '
        #                   'should be defined. Found `None`.')
        #self.input_spec = base.InputSpec(min_ndim=2,
        #                                 axes={-1: input_shape[-1].value})
        #self.kernel = self.add_variable('kernel',
        #                                shape=[input_shape[-1].value, self.units],
        #                                initializer=self.kernel_initializer,
        #                                regularizer=self.kernel_regularizer,
        #                                dtype=self.dtype,
        #                                trainable=True)
        #if self.use_bias:
        #  self.bias = self.add_variable('bias',
        #                                shape=[self.units,],
        #                                initializer=self.bias_initializer,
        #                                regularizer=self.bias_regularizer,
        #                                dtype=self.dtype,
        #                                trainable=True)
        #else:
        #  self.bias = None
        self.built = True


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
        "kernel_regularizer": _default_regularizer_hparams(),
        "bias_regularizer": _default_regularizer_hparams(),
        "activity_regularizer": _default_regularizer_hparams(),
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
    })
    return kwargs

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
    """Returns the default keyword argument values of dense layer
    defined in :tf_main:`tf.layers.Dense <layers/Dense>`.
    """
    kwargs = _common_default_conv_dense_kwargs()
    kwargs.update({
        "units": 256
    })
    return kwargs

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
                - User-defined layer class in :mod:`texar.custom`. The class \
                  must inherit :tf_main:`Layer <layers/Layer>`.
                - External layer. Must provide the full path, \
                  e.g., :attr:`"my_module.MyInitializer"`, or the instance.

            "kwargs" : dict
                A dictionary of arguments for constructor of the
                layer class. Ignored if :attr:`"type"` is a layer instance.

                - Arguments named "activation" have values of type `str` that \
                specify the name or full path to the activation function. \
                The activation function will be used for making the layer.
                - Arguments named "*_regularizer" and "*_initializer" \
                have values of type `dict` that specifiy hyperparameters of \
                respective regularizers and initializers. Regularizer and \
                initializer instances will be created accordingly and used for \
                making the layer.
                - Arguments named "*_constraint" have values of type `str` \
                that specify the name or full path to the constant function. \
                The constraint function will be used.

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
        layer_modules = ["tensorflow.layers", "texar.costum"]
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


def sinusoid_positional_encoding(inputs,
                                 num_units,
                                 scale=False,
                                 reuse=None,
                                 max_time=None,
                                 variable_scope='sinuoid_positional_embedding'):
    """obtain a positional encoding of inputs
    Args:
        inputs: [Tensor] A Tensor of shape `[batch_size, max_time, hidden_dim]`
        max_time: [Int], max time steps
        hidden_dim: [Int], hidden size of embedding
        scale: [Boolean], If True, the output will be multiplied by sqrt(num_units)
        variable_scope: [String], Optional scope for 'variable_scope'
    """
    batch_size = inputs.shape.as_list()[0]
    dynamic_max_time = tf.shape(inputs)[1]
    hidden_dim = num_units
    with tf.variable_scope(variable_scope, reuse=reuse):
        position_idx = tf.tile(tf.expand_dims(tf.range(max_time), 0), [batch_size, 1])
        position_enc = np.array([
            [pos /np.power(10000, 2.*i/hidden_dim) for i in range(hidden_dim)]
            for pos in range(max_time)])

        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])

        lookup_table = tf.convert_to_tensor(position_enc, dtype=tf.float32)
        outputs = tf.nn.embedding_lookup(lookup_table, position_idx)
        if scale:
            outputs = outputs * hidden_dim**0.5
        outputs = outputs[:, :dynamic_max_time, :]
        return outputs

def multihead_attention(queries,
                        keys,
                        num_units=None,
                        num_heads=8,
                        dropout_rate=0,
                        causality=False,
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

        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)

        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))

        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        #not sure why there should be key_masks and query_masks
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))
        key_masks = tf.tile(key_masks, [num_heads, 1])
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])

        paddings = tf.ones_like(outputs)*(-2**32+1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)

        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])
            tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense()
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])

            paddings = tf.ones_like(masks)*(-2**32+1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)

        outputs = tf.nn.softmax(outputs)

        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))
        query_masks = tf.tile(query_masks, [num_heads, 1])
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])
        outputs *= query_masks

        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=context.is_train())

        outputs = tf.matmul(outputs, V_)
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)
        #(batch_size, length_query, attention_size)

        #residual connection
        if num_units == queries.get_shape().as_list()[-1]:
            outputs += queries

        outputs = normalize(outputs)

    return outputs


def normalize(inputs,
              epsilon=1e-8,
              scope='ln',
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
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        outputs = tf.nn.batch_normalization(inputs, mean, variance,\
            offset=beta, scale=gamma, variance_epsilon=epsilon)
    return outputs
