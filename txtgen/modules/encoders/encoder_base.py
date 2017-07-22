#
"""
Base class for encoders
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from txtgen.modules.module_base import ModuleBase


def make_encoder_output(outputs, state, **kwargs):
  """Wraps the encoder outputs with a dictionary.

  Args:
    outputs: Outputs of the encoder.
    state: Final state of the encoder.
    **kwargs: Other outputs.

  Returns:
    A dictionary containing the outputs.
  """
  return dict({"outputs": outputs, "state": state}, **kwargs)


class EncoderBase(ModuleBase):
  """Base class inherited by all encoder classes.
  """

  def __init__(self, name, hparams=None):
    ModuleBase.__init__(name, hparams)

  def _build(self, *args, **kwargs):
    return self.encode(*args, **kwargs)

  def encode(self, inputs, *args, **kwargs):
    """Performs the encoder logic.

    Args:
      inputs: Inputs to the encoder.
      *args: Other arguments.
      **kwargs: Keyword arguments.

    Returns:
      A dictionary of outputs.
    """
    raise NotImplementedError

