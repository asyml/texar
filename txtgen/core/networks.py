#
"""
Various neural networks
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
from txtgen.core.layers import get_rnn_cell


def get_forward_rnn(cell_hparams, inputs, **kwargs):
    """Returns the results of a forward rnn with the specified cell

    Args:
      cell_hparams: A dictionary of cell hyperparameters
      inputs: Inputs of the RNN
      **kwargs: Keyword arguments of the RNN

    Returns:
      Outputs and the final state of the RNN
    """
    cell = get_rnn_cell(cell_hparams)
    return tf.nn.dynamic_rnn(
        cell=cell,
        inputs=inputs,
        dtype=tf.float32,
        **kwargs)
