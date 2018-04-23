#
"""
Various convolutional network encoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from texar.modules.encoders.encoder_base import EncoderBase
from texar.modules.networks.conv_networks import Conv1DNetwork

__all__ = [
    "Conv1DEncoder"
]

class Conv1DEncoder(Conv1DNetwork, EncoderBase):
    """Simple Conv-1D encoder which consists of a sequence of conv layers
    followed with a sequence of dense layers.

    Wraps :class:`~texar.modules.Conv1DNetwork` to be a subclass of
    :class:`~texar.modules.EncoderBase`. Has exact the same functionality
    with `Conv1DNetwork`.
    """

    def __init__(self, hparams=None): # pylint: disable=super-init-not-called
        Conv1DNetwork.__init__(self, hparams)

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        The same as :meth:`texar.modules.Conv1DNetwork.default_hparams`,
        except that the default name is 'conv_encoder'.
        """
        hparams = Conv1DNetwork.default_hparams()
        hparams['name'] = 'conv_encoder'
        return hparams

