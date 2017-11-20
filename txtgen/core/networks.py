#
"""
Various neural networks and related utilities.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from txtgen.core import layers
from txtgen.core import utils


class FeedForwardNetwork(object):
    """Feed forward neural network that consists of a sequence of layers.
    """

    def __init__(self, hparams=None):
        raise NotImplementedError

    @staticmethod
    def default_hparams():
        return {
            "layers": []
        }

    #def build(self):


