#
"""
Various neural network layers
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import tensorflow.contrib.rnn as rnn

from txtgen.hyperparams import HParams
from txtgen.core.utils import get_instance, switch_dropout


def default_rnn_cell_hparams():
    """Returns default hyperparameters of an RNN cell.

    Returns:
        dict: A dictionary with the following structure and values:

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


def get_rnn_cell(hparams):
    """Creates an RNN cell.

    Args:
        hparams (dict or HParams): Cell hyperparameters.

    Returns:
        An instance of `RNNCell`.
    """

    d_hp = hparams["dropout"]
    if d_hp["variational_recurrent"] and \
            len(d_hp["input_size"]) != hparams["num_layers"]:
        raise ValueError(
            "If variational_recurrent=True, input_size must be a list of "
            "num_layers(%d) integers. Got len(input_size)=%d." %
            (hparams["num_layers"], len(d_hp["input_size"])))

    cells = []
    cell_kwargs = hparams["cell"]["kwargs"]
    if isinstance(cell_kwargs, HParams):
        cell_kwargs = cell_kwargs.todict()
    for layer_i in range(hparams["num_layers"]):
        # Create the basic cell
        cell_type = hparams["cell"]["type"]
        cell_modules = ['txtgen.custom', 'tensorflow.contrib.rnn']
        cell = get_instance(cell_type, cell_kwargs, cell_modules)

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
                input_keep_prob=switch_dropout(d_hp["input_keep_prob"]),
                output_keep_prob=switch_dropout(d_hp["output_keep_prob"]),
                state_keep_prob=switch_dropout(d_hp["state_keep_prob"]),
                **vr_kwargs)

        # Optionally add residual and highway connections
        if layer_i > 0:
            if hparams["residual"]:
                cell = rnn.ResidualWrapper(cell) # pylint: disable=redefined-variable-type
            if hparams["highway"]:
                cell = rnn.HighwayWrapper(cell)

        cells.append(cell)

    if hparams["num_layers"] > 1:
        cell = rnn.MultiRNNCell(cells) # pylint: disable=redefined-variable-type
    else:
        cell = cells[0]

    return cell


def default_embedding_hparams():
    """Returns default hyperparameters of embedding.

    Returns:
        dict: A dictionary with the following structure and values:

        .. code-block:: python

            {
                "name": "embedding", # A string. Name of the embedding variable.
                "dim": 100,          # An integer. Embedding dimension.
                "initializer": {     # Initializer of embedding values.
                    # A string. Name or full path to the initializer class.
                    # An initializer is a class inheriting from
                    # `tensorflow.Initializer`, which can be built-in
                    # classes in module `tensorflow`, or user-defined
                    # classes in `txtgen.custom`, or a full path like
                    # `my_module.MyInitializer`.
                    "type": "tensorflow.random_uniform_initializer",

                    # A dictionary of arguments for constructor of the
                    # initializer class. An initializer is created by
                    # calling `initialzier_class(**kwargs)` where
                    # `initializer_class` is specified in `type`.
                    "kwargs": {
                        "minval": -0.1,
                        "maxval": 0.1,
                        "seed": None
                    }
                }
            }
    """
    return { #TODO(zhiting): allow more hparams like regularizer
        "name": "embedding",
        "dim": 100,
        "initializer": {
            "type": "tensorflow.random_uniform_initializer",
            "kwargs": {
                "minval": -0.1,
                "maxval": 0.1,
                "seed": None
            }
        }
    }


def get_embedding(hparams,
                  init_values=None,
                  vocab_size=None,
                  trainable=True,
                  variable_scope=None):
    """Creates embedding variable if not exists.

    Args:
        hparams (dict or HParams): Embedding hyperparameters. See
            :meth:`~txtgen.core.layers.default_embedding_hparams` for the
            default values. If :attr:`init_values` is given,
            :attr:`hparams["initializer"]`, and :attr:`hparams["dim"]` are
            ignored.
        init_values (Tensor or numpy array, optional): Initial values of the
            embedding variable. If not given, embedding is initialized as
            specified in :attr:`hparams["initializer"]`.
        vocab_size (int, optional): The vocabulary size. Required if
            :attr:`init_values` is not provided.
        trainable (bool): If `True` (default), the embedding variable is a
            trainable variable.
        variable_scope (string or VariableScope, optional): Variable scope of
            the embedding variable.

    Returns:
        Variable: A 2D `Variable` of the same shape with :attr:`init_values`
        or of the shape [:attr:`vocab_size`, :attr:`hparams["dim"]`].
    """
    with tf.variable_scope(variable_scope, "embedding"): # pylint: disable=not-context-manager
        if init_values is None:
            kwargs = hparams["initializer"]["kwargs"]
            if isinstance(kwargs, HParams):
                kwargs = kwargs.todict()
            initializer = get_instance(hparams["initializer"]["type"],
                                       kwargs,
                                       ["txtgen.custom", "tensorflow"])
            return tf.get_variable(name=hparams["name"],
                                   shape=[vocab_size, hparams["dim"]],
                                   initializer=initializer,
                                   trainable=trainable)
        else:
            return tf.get_variable(name=hparams["name"],
                                   initializer=init_values,
                                   trainable=trainable)

