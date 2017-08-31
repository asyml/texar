#
"""
Various neural network layers
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow.contrib.rnn as rnn

from txtgen.core.utils import get_instance, switch_dropout


def default_rnn_cell_hparams():
    """Returns default hyperparameters of an RNN cell.

    Returns:
        A dictionary of default hyperparameters of an RNN cell, with the
        following structure and values:

        ```python
        {
            "cell": {
                # Name or full path of the cell class. E.g., the classname of
                # built-in cells in `tensorflow.contrib.rnn`, or the classname
                # of user-defined cells in `txtgen.custom`, or a full path
                # like "my_module.MyCell".

                "type": "BasicLSTMCell",

                # A dictionary of arguments for constructor of the cell class.
                # An RNN cell is created by calling the cell class named in
                # `type` passing the arguments specified in `kwargs` as in:
                #     cell_class(**kwargs)

                "kwargs": {
                    "num_units": 64
                }
            },

            # Dropout applied to the cell if `use=True`. If `num_layers>1`,
            # dropout is applied to the cell of each layer independently. See
            # `tensorflow.contrib.rnn.DropoutWrapper` for each of the
            # hyperparameters.

            "dropout": {
                "use": False,
                "input_keep_prob": 1.0,
                "output_keep_prob": 1.0,
                "state_keep_prob": 1.0,
                "variational_recurrent": False
            },

            "num_layers": 1       # Number of cell layers

            # Whether to apply residual connection on cell inputs and outputs.
            # If `num_layers>1`, the connection is between the inputs and
            # outputs of the multi-layer cell as a whole

            "residual": False,

            # Whether to apply highway connection on cell inputs and outputs. If
            # `num_layers>1`, the connection is between the inputs and outputs
            # of the multi-layer cell as a whole

            "highway": False,
        }

        ```
    """
    return {
        "cell": {
            "type": "BasicLSTMCell",
            "kwargs": {
                "num_units": 64
            }
        },
        "dropout": {
            "use": False,
            "input_keep_prob": 1.0,
            "output_keep_prob": 1.0,
            "state_keep_prob": 1.0,
            "variational_recurrent": False
        },
        "num_layers": 1,
        "residual": False,
        "highway": False,
    }


def get_rnn_cell(cell_hparams):
    """Creates an RNN cell.

    Args:
      cell_hparams: a dictionary of hyperparameters.

    Returns:
      An instance of RNN cell.
    """
    cells = []
    for _ in range(cell_hparams["num_layers"]):
        cell_type = cell_hparams["cell"]["type"]
        cell_modules = ['txtgen.custom', 'tensorflow.contrib.rnn']
        cell = get_instance(cell_type, cell_hparams["cell"]["args"],
                            cell_modules)

        d_hp = cell_hparams["dropout"]
        if d_hp["use"]:
            cell = rnn.DropoutWrapper(
                cell=cell,
                input_keep_prob=switch_dropout(d_hp["input_dropout_prob"]),
                output_keep_prob=switch_dropout(d_hp["output_dropout_prob"]),
                state_keep_prob=switch_dropout(d_hp["state_dropout_prob"]),
                variational_recurrent=d_hp["variational_recurrent"])

        cells.append(cell)

    if cell_hparams["num_layers"] > 1:
        cell = rnn.MultiRNNCell(cells)    # pylint: disable=redefined-variable-type
    else:
        cell = cells[0]

    if cell_hparams["residual"]:
        cell = rnn.ResidualWrapper(cell)
    if cell_hparams["highway"]:
        cell = rnn.HighwayWrapper(cell)

    return cell

