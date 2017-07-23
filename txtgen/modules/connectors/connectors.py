#
"""
Various connectors
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from txtgen.modules.connectors.connector_base import ConnectorBase


class ForwardConnector(ConnectorBase):
    """Simply forward the final state of encoder to decoder without
    transformation.

    The state structures and sizes of the encoder and decoder must be the same.
    """

    def __init__(self, name="forward_connector"):
        ConnectorBase.__init__(self, name, None)

    def connect(self, encoder_outputs, *args, **kwargs):
        raise NotImplementedError
        # return make_connector_outputs(encoder_outputs["state"])

    @staticmethod
    def default_hparams():
        return {}


class StochasticConnector(ConnectorBase):
    """Samples decoder initial state from a distribution defined by the
    encoder outputs.

    Used in, e.g., variational autoencoders, adversarial autoencoders, and other
    models.
    """
    pass  # TODO
