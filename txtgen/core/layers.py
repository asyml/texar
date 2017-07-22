#
"""
Various neural network layers
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow.contrib.rnn as rnn

from txtgen.core.utils import get_instance, switch_dropout
from txtgen import custom


def default_rnn_cell_hparams():
    """Returns default hyperparameters of an RNN cell.

    Returns:
      A dictionary of default hyperparameters of an RNN cell, with the following
      structure and values:

      ```python
      {
        "cell": {
            # Class name of the cell, either defined by users in `txtgen.custom`,
            # or built-in class pre-defined in `tensorflow.contrib.rnn`

            "type": "BasicLSTMCell",

            # A dictionary of arguments for the constructor of the cell class

            "args": {
              "num_units": 64
            }
        },

        # Dropout applied to the cell. If `num_layers>1`, dropout is applied to
        # the cell of each layer. See `tensorflow.contrib.rnn.DropoutWrapper`
        # for each of the hyperparameters

        "dropout": {
            "use": False,
            "input_keep_prob": 1.0,
            "output_keep_prob": 1.0,
            "state_keep_prob": 1.0,
            "variational_recurrent": False
        },
        "num_layers": 1       # Number of cell layers

        # Whether to apply residual connection on cell inputs and outputs. If
        # `num_layers>1`, the connection is between the inputs and outputs of the
        # multi-layer cell as a whole

        "residual": False,

        # Whether to apply highway connection on cell inputs and outputs. If
        # `num_layers>1`, the connection is between the inputs and outputs of the
        # multi-layer cell as a whole

        "highway": False,
      }

      ```
    """
    return {
        "cell": {
            "type": "BasicLSTMCell",
            "args": {
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
    """Creates an RNN cell

    Args:
      cell_hparams: a dictionary of hyperparameters

    Returns:
      An instance of RNN cell
    """
    cells = []
    for _ in range(cell_hparams["num_layers"]):
        cell_type = cell_hparams["cell"]["type"]
        try:
            try:
                cell_class = getattr(custom, cell_type)
            except:
                cell_class = getattr(rnn, cell_type)
        except:
            raise ValueError("Cell type not found: %s" % cell_type)
        cell = get_instance(cell_class, cell_hparams["cell"]["args"])

        dpt_hparams = cell_hparams["dropout"]
        if dpt_hparams["use"]:
            cell = rnn.DropoutWrapper(
                cell=cell,
                input_keep_prob=switch_dropout(dpt_hparams["input_dropout_prob"]),
                output_keep_prob=switch_dropout(dpt_hparams["output_dropout_prob"]),
                state_keep_prob=switch_dropout(dpt_hparams["state_dropout_prob"]),
                variational_recurrent=dpt_hparams["variational_recurrent"])

        cells.append(cell)

    if cell_hparams["num_layers"] > 1:
        cell = rnn.MultiRNNCell(cells)
    else:
        cell = cells[0]

    if cell_hparams["residual"]:
        cell = rnn.ResidualWrapper(cell)
    if cell_hparams["highway"]:
        cell = rnn.HighwayWrapper(cell)   # FIXME No Highway wrapper yet

    return cell
