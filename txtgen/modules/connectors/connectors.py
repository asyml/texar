#
"""
Various connectors
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

from txtgen.modules.connectors.connector_base import ConnectorBase

class ForwardConnector(ConnectorBase):
  """Simply forward the final state of encoder to decoder without
  transformation.

  The state structures and sizes of the encoder and decoder must be the same.
  """
  def __init__(self, name="forward_connector"):
    ModuleBase.__init__(name, None)

  def connect(self, encoder_outputs):
    return make_connector_ouputs(encoder_outputs["state"])

  @staticmethod
  def default_hparams(self):
    return {}


class StochasticConnector(ConnectorBase):
  """Samples decoder initial state from a distribution defined by the
  encoder outputs.

  Used in, e.g., variational autoencoders, adversarial autoencoders, and other
  models.
  """
  pass #TODO
