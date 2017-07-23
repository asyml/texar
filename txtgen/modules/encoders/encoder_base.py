#
"""
Base class for encoders
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from txtgen.modules.module_base import ModuleBase


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

