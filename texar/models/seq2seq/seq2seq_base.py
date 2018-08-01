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
                 source_vocab,
                 target_vocab=None,
                 source_embedder=None,
                 target_embedder=None,
                 encoder=None,
                 decoder=None,
                 connector=None,
                 hparams=None):
        ModelBase.__init__(hparams)

        self._src_vocab = source_vocab
        if source_embedder is not None:
            if self._src_vocab.size != source_embedder.num_embeds:
                raise ValueError(
                    'source vocab size ({}) does not match the source embedder '
                    'size ({}).'.format(self._src_vocab.size,
                                        source_embedder.num_embeds))

        self._tgt_vocab = target_vocab or self._src_vocab
        if target_embedder is not None:
            if self._tgt_vocab.size != target_embedder.num_embeds:
                raise ValueError(
                    'target vocab size ({}) does not match the target embedder '
                    'size ({}).'.format(self._tgt_vocab.size,
                                        target_embedder.num_embeds))

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
            "decoding_strategy_train": "train_greedy",
            "decoding_strategy_infer": "infer_greedy",
            "beam_search_width": 0,
            "connector_type": "MLPTransformConnector",
            "connector": {},
            "optimization": {}
        })
        return hparams

    def _build_embedders(self):
        if self._src_embedder is None:
            kwargs = {
                "vocab_size": self._src_vocab.size,
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
                    "vocab_size": self._tgt_vocab.size,
                }
                if self._hparams.embedder_hparams_share:
                    kwargs["hparams"] = self._hparams.source_embedder.todict()
                else:
                    kwargs["hparams"] = self._hparams.target_embedder.todict()
                self._tgt_embedder = utils.check_or_get_instance(
                    self._hparams.target_embedder_type, kwargs,
                    ["texar.modules, texar.custom"])

    def _build_encoder(self):
        if self._encoder is None:
            kwargs = {
                "hparams": self._hparams.encoder.todict()
            }
            self._encoder = utils.check_or_get_instance(
                self._hparams.encoder_type, kwargs,
                ["texar.modules, texar.custom"])

    def _build_decoder(self):
        raise NotImplementedError

    def _build_connector(self):
        if self._connector is None:
            kwargs = {
                "output_size": self._decoder.state_size,
                "hparams": self._hparams.connector.todict()
            }
            self._connector = utils.check_or_get_instance(
                self._hparams.connector_type, kwargs,
                ["texar.modules, texar.custom"])

