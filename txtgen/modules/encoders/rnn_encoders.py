#
"""
Various RNN encoders
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from txtgen.modules.encoders.encoder_base import EncoderBase
from txtgen.modules.encoders.encoder_base import make_encoder_output
from txtgen.core.layers import default_rnn_cell_hparams
from txtgen.core import networks


class ForwardRNNEncoder(EncoderBase):
  """One directional forward RNN encoder
  """

  def __init__(self, name="forward_rnn_encoder", hparams=None):
    """Initializes the encoder.

    Args:
      name: Name of the encoder.
      hparams: A dictionary of hyperparameters with the following structure and
        default values:

        ```python
        {
          # A dictionary of hyperparameters of the rnn cell. See
          # `txtgen.modules.encoders.encoder_base.make_encoder_output` for the
          # structure.

          "rnn_cell": default_rnn_cell_hparams
        }
        ```
    """
    ModuleBase.__init__(name, hparams)

  def encode(self, inputs, **kwargs):
    outputs, state = networks.get_forward_rnn(
        self.hparams.rnn_cell, inputs, **kwargs)
    return make_encoder_output(outputs, state)

  @staticmethod
  def default_hparams():
    return {
        "rnn_cell": default_rnn_cell_hparams()
    }
