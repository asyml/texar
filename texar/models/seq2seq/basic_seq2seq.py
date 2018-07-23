#
"""
The basic seq2seq model without attention.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from texar.models.seq2seq.seq2seq_base import Seq2seqBase

__all__ = [
    "BasicSeq2seq"
]

class BasicSeq2seq(Seq2seqBase):
    """The basic seq2seq model without attention.
    """

    def __init__(self, hparams=None):
        Seq2seqBase.__init__(hparams)

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.
        """
        hparams = {
            "name": "basic_seq2seq"
        }
        return hparams

