#
"""
Base class for encoders
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from txtgen.modules.module_base import ModuleBase
from txtgen.core.layers import default_rnn_cell_hparams


class EncoderBase(ModuleBase):
    """Base class inherited by all encoder classes.
    """

    def __init__(self, name, hparams=None):
        ModuleBase.__init__(name, hparams)

    def _build(self, inputs, *args, **kwargs):
        """Encodes the inputs.

        Args:
          inputs: Inputs to the encoder.
          *args: Other arguments.
          **kwargs: Keyword arguments.

        Returns:
          Encoding results.
        """
        raise NotImplementedError

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        The dictionary has the following structure and default values:

            ```python
            {
              # A dictionary of rnn cell hyperparameters. See
              # `txtgen.core.layers.default_rnn_cell_hparams` for the
              # structure and default values.

              "rnn_cell": default_rnn_cell_hparams
            }
            ```
        """
        return {
            "rnn_cell": default_rnn_cell_hparams()
        }


