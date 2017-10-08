#
"""
Base class for connectors that transform results of other modules (e.g., the
final state of an encoder) into decoder initial states.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from txtgen.modules.module_base import ModuleBase


class ConnectorBase(ModuleBase):
    """Base class inherited by all connector classes. A connector is to
    transform outputs of other modules into decoder initial states.

    Args:
        decoder_state_size: Size of state of the decoder cell. Can be an
            Integer, a Tensorshape , or a tuple of Integers or TensorShapes.
            This can typically be obtained by `decoder.cell.state_size`.
        hparams (dict): Hyperparameters of connector.
    """

    def __init__(self, decoder_state_size, hparams=None):
        ModuleBase.__init__(self, hparams)
        self._decoder_state_size = decoder_state_size

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.
        """
        return {
            "name": "connector"
        }

    def _build(self, *args, **kwargs):
        """Transforms inputs to the decoder initial states.
        """
        raise NotImplementedError
