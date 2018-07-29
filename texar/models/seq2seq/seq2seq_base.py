#
"""
Base class for seq2seq models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from texar.models.model_base import ModelBase
from texar.utils import utils

__all__ = [
    "Seq2seqBase"
]

class Seq2seqBase(ModelBase):
    """Base class inherited by all seq2seq model classes.
    """

    def __init__(self,
                 source_vocab_size=None,
                 target_vocab_size=None,
                 source_embedder=None,
                 target_embedder=None,
                 encoder=None,
                 decoder=None,
                 connector=None,
                 hparams=None):
        ModelBase.__init__(hparams)

        self._src_vocab_size = source_vocab_size
        if source_vocab_size is None:
            if source_embedder is None:
                raise ValueError('Either `source_vocab_size` or '
                                 '`source_embedder` must be given.')
            self._src_vocab_size = source_embedder.num_embeds

        self._tgt_vocab_size = target_vocab_size
        if target_vocab_size is None:
            if target_embedder is not None:
                self._tgt_vocab_size = target_embedder.num_embeds
            else:
                self._tgt_vocab_size = self._src_vocab_size

        self._src_embedder = source_embedder
        self._tgt_embedder = target_embedder
        self._encoder = encoder
        self._decoder = decoder
        self._connector = connector

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.
        """
        hparams = ModelBase.default_hparams()
        hparams.update({
            "name": "seq2seq",
            "source_embedder_type": "WordEmbedder",
            "source_embedder": {},
            "target_embedder_type": "WordEmbedder",
            "target_embedder": {},
            "embedder_share": True,
            "embedder_hparams_share": True,
            "encoder_type": "UnidirectionalRNNEncoder",
            "encoder": {},
            "decoder_type": "BasicRNNDecoder",
            "decoder": {},
            "connector_type": "MLPTransformConnector",
            "connector": {},
            "optimization": {}
        })
        return hparams

    def _get_embedders(self):
        if self._src_embedder is None:
            kwargs = {
                "vocab_size": self._src_vocab_size,
                "hparams": self._hparams.source_embedder.todict()
            }
            self._src_embedder = utils.check_or_get_instance(
                self._hparams.source_embedder_type, kwargs,
                ["texar.modules, texar.custom"])

        if self._tgt_embedder is None:
            if self._hparams.embedder_share:
                self._tgt_embedder = self._src_embedder
            else:
                kwargs = {
                    "vocab_size": self._tgt_vocab_size,
                }
                if self._hparams.embedder_hparams_share:
                    kwargs["hparams"] = self._hparams.source_embedder.todict()
                else:
                    kwargs["hparams"] = self._hparams.target_embedder.todict()
                self._tgt_embedder = utils.check_or_get_instance(
                    self._hparams.target_embedder_type, kwargs,
                    ["texar.modules, texar.custom"])

    def _get_encoder(self):
        if self._encoder is None:
            kwargs = {
                "hparams": self._hparams.encoder.todict()
            }
            self._encoder = utils.check_or_get_instance(
                self._hparams.encoder_type, kwargs,
                ["texar.modules, texar.custom"])

    def _get_decoder(self):
        raise NotImplementedError

    def _get_connector(self):
        if self._connector is None:
            kwargs = {
                "output_size": self._decoder.state_size,
                "hparams": self._hparams.connector.todict()
            }
            self._connector = utils.check_or_get_instance(
                self._hparams.connector_type, kwargs,
                ["texar.modules, texar.custom"])

