#
"""
The basic seq2seq model without attention.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from texar.models.seq2seq.seq2seq_base import Seq2seqBase
from texar.utils import utils
from texar.utils.shapes import get_batch_size

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

    def embed_source(self, features, labels, mode):
        """Embeds the inputs.
        """
        return self._src_embedder(ids=features["source_text_ids"], mode=mode)

    def encode(self, features, labels, mode):
        """Encodes the inputs.
        """
        embedded_source = self.embed_source(features, labels, mode)
        outputs, final_state = self._encoder(
            embedded_source,
            sequence_length=features["source_length"],
            mode=mode)
        return outputs, final_state

    def embed_target(self, features, labels, mode):
        """Embeds the target inputs. Used in training.
        """
        return self._tgt_embedder(ids=features["target_text_ids"], mode=mode)

    def _connect(self, encoder_results, features, labels, mode):
        """Transforms encoder final state into decoder initial state.
        """
        enc_state = encoder_results["final_state"]
        possible_kwargs = {
            "inputs": enc_state,
            "batch_size": get_batch_size(enc_state)
        }
        outputs = utils.call_function_with_redundant_kwargs(
            self._connector, possible_kwargs)
        return outputs

    def decode(self, encoder_results, features, labels, mode):
        """Decodes.
        """


    def _build(self, features, labels, params, mode, config=None):
        self._get_embedders()
