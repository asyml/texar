#
"""
Base class for encoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from texar.module_base import ModuleBase

__all__ = [
    "EncoderBase"
]

class EncoderBase(ModuleBase):
    """Base class inherited by all encoder classes.
    """

    def __init__(self, hparams=None):
        ModuleBase.__init__(self, hparams)

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.
        """
        return {
            "name": "encoder"
        }

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

