#
"""
Base class for connectors that transform encoder outputs into decoder initial states.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from txtgen.modules.module_base import ModuleBase


def make_connector_output(outputs, **kwargs):
    """Wraps the connector outputs with a dictionary.

    Args:
      outputs: Outputs of the connector.
      **kwargs: Other outputs.

    Returns:
      A dictionary containing the outputs.
    """
    return dict({"outputs": outputs}, **kwargs)


class ConnectorBase(ModuleBase):
    """Base class inherited by all connector classes.
    """

    @staticmethod
    def default_hparams():
        pass

    def __init__(self, name, hparams=None):
        ModuleBase.__init__(name, hparams)

    def _build(self, *args, **kwargs):
        return self.connect(*args, **kwargs)

    def connect(self, encoder_outputs, *args, **kwargs):
        """Transforms the outputs of encoder to the initial states of decoder.

        Args:
          encoder_outputs: A dictionary of encoder outputs. See
            `txtgen.modules.encoder.encoder_base.make_encoder_output`.
          *args: Other arguments.
          **kwargs: Keyword arguments.

        Returns:
          A dictionary of outputs.
        """
        raise NotImplementedError
