#
"""
Various neural network layers
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import tensorflow.contrib.rnn as rnn

from txtgen.core.utils import get_instance, switch_dropout


def default_rnn_cell_hparams():
    """Returns default hyperparameters of an RNN cell.
    
    Returns:
        A dictionary of default hyperparameters of an RNN cell, with the
        following structure and values:
        
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
    
                "num_layers": 1       # Number of cell layers
    
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
    """
    
    Creates an RNN cell.
       
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
    for layer_i in range(hparams["num_layers"]):
        # Create the basic cell
        cell_type = hparams["cell"]["type"]
        cell_modules = ['txtgen.custom', 'tensorflow.contrib.rnn']
        cell = get_instance(cell_type, hparams["cell"]["kwargs"],
                            cell_modules)

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
