#
"""
Base class for connectors that transform encoder outputs/states into decoder
initial states.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from txtgen.modules.module_base import ModuleBase


class ConnectorBase(ModuleBase):
    """Base class inherited by all connector classes. A connector is to
    transform encoder outputs and states into decoder initial states.
    """

    def __init__(self, decoder_state_size, hparams=None, name="connector"):
        """Initializes the connector.

        Args:
            decoder_state_size: Size of state of the decoder cell. Can be an
                Integer, a Tensorshape , or a tuple of Integers or TensorShapes.
                This can typically be obtained by `decoder.cell.state_size`.
            hparams: A dictionary of hyperparameters.
            name: Name of connector.
        """
        super(ConnectorBase, self).__init__(name, hparams)
        self._decoder_state_size = decoder_state_size

    def _build(self, *args, **kwargs):
        """Transforms the encoder outputs and states to the decoder initial
        states.
        """
        raise NotImplementedError
