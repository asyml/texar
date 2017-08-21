#
"""
Various RNN encoders
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from txtgen.modules.encoders.encoder_base import EncoderBase
from txtgen.core.layers import get_rnn_cell
from txtgen.core.layers import default_rnn_cell_hparams


class ForwardRNNEncoder(EncoderBase):
    """One directional forward RNN encoder.
    """

    def __init__(self, cell=None, hparams=None, name="forward_rnn_encoder"):
        """Initializes the encoder.

        Args:
            cell: (optional) An instance of `RNNCell`. If it is not specified,
                a cell is created as specified by `rnn_cell` in `hparams`.
            hparams: (optional) A dictionary of hyperparameters. If it is not
                specified, the default hyperparameter setting is used. See
                `default_hparams` for the sturcture and default values.
            name: Name of the encoder.
        """
        EncoderBase.__init__(self, hparams, name)
        if cell is not None:
            self._cell = cell
        else:
            self._cell = get_rnn_cell(self.hparams.rnn_cell)

    def _build(self, inputs, **kwargs):
        """Encodes the inputs.

        Args:
            inputs: Input sequences to encode.
            **kwargs: Optional keyword arguments of `tensorflow.nn.dynamic_rnn`,
                such as `sequence_length`, `initial_state`, etc.

        Returns:
            Outputs and final state of the encoder.
        """
        if ('dtype' not in kwargs) and ('initial_state' not in kwargs):
            return tf.nn.dynamic_rnn(
                cell=self._cell,
                inputs=inputs,
                dtype=tf.float32,
                **kwargs)
        else:
            return tf.nn.dynamic_rnn(
                cell=self._cell,
                inputs=inputs,
                **kwargs)

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        The dictionary has the following structure and default values:

            ```python
            {
              # A dictionary of rnn cell hyperparameters. See
              # `txtgen.core.layers.default_rnn_cell_hparams` for the
              # structure and default values. It is not used if a cell instance
              # is already specified.

              "rnn_cell": default_rnn_cell_hparams
            }
            ```
        """
        return {
            "rnn_cell": default_rnn_cell_hparams()
        }


