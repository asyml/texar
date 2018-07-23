#
"""
Base class for seq2seq models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from texar.models.model_base import ModelBase

__all__ = [
    "Seq2seqBase"
]

class Seq2seqBase(ModelBase):
    """Base class inherited by all seq2seq model classes.
    """

    def __init__(self, hparams=None):
        ModelBase.__init__(hparams)

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.
        """
        hparams = {
            "name": "seq2seq"
        }
        return hparams

    def _get_encoder(self):
        raise NotImplementedError

    def _get_decoder(self):
        raise NotImplementedError

    def _get_connector(self):
        raise NotImplementedError
