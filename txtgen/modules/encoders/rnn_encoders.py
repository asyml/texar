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


class ForwardRNNEncoder(EncoderBase):
    """One directional forward RNN encoder.
    """

    def __init__(self, name="forward_rnn_encoder", hparams=None):
        """Initializes the encoder.

        Args:
            name: Name of the encoder.
            hparams: A dictionary of hyperparameters. See `default_hparams` for
                the sturcture and default values.
        """
        EncoderBase.__init__(self, name, hparams)
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

