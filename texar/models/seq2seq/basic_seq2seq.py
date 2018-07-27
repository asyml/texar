#
"""
The basic seq2seq model without attention.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from texar.models.seq2seq.seq2seq_base import Seq2seqBase
from texar.utils import utils

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
        hparams = Seq2seqBase.default_hparams()
        hparams.update({
            "name": "basic_seq2seq"
        })
        return hparams

    def _get_decoder(self):
        if self._decoder is None:
            kwargs = {
                "vocab_size": self._tgt_vocab_size,
                "hparams": self._hparams.decoder.todict()
            }
            self._decoder = utils.check_or_get_instance(
                self._hparams.decoder_type, kwargs,
                ["texar.modules, texar.custom"])

    def embed(self, features, labels):
        """Embeds the inputs.
        """

    #def encode(self, features, labels):
    #    """Encodes the inputs.
    #    """
    #    outputs, final_state = self._encoder(
    #        self._src_embedder())

    def _build(self, features, labels, params, mode, config=None):
        self._get_embedders()
