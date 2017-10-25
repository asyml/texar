#
"""
Base class for connectors that transform inputs into specified output shape.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from txtgen.modules.module_base import ModuleBase

__all__ = [
    "ConnectorBase"
]

class ConnectorBase(ModuleBase):
    """Base class inherited by all connector classes. A connector is to
    transform inputs into outputs with any specified shape, e.g., transforms
    the final state of an encoder to the initial state of a decoder.

    Args:
        output_size: Size of output. Can be an int, a tuple of int, a
            Tensorshape, or a tuple of TensorShapes. For example, to transform
            to decoder state size, set `output_size=decoder.cell.state_size`.
        hparams (dict): Hyperparameters of connector.
    """

    def __init__(self, output_size, hparams=None):
        ModuleBase.__init__(self, hparams)
        self._output_size = output_size

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.
        """
        return {
            "name": "connector"
        }

    def _build(self, *args, **kwargs):
        """Transforms inputs to outputs with specified shape.
        """
        raise NotImplementedError
