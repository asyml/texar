#
"""
The basic seq2seq model without attention.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from texar.models.seq2seq.seq2seq_base import Seq2seqBase
from texar.utils import utils
from texar.utils.shapes import get_batch_size
from texar.modules.decoders.beam_search_decode import beam_search_decode

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

    def _build_decoder(self):
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

        return {'outputs': outputs, 'final_state': final_state}

    def embed_target(self, features, labels, mode):
        """Embeds the target inputs. Used in training.
        """
        return self._tgt_embedder(ids=labels["target_text_ids"], mode=mode)

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

    def _decode_train(self, initial_state, encoder_results, features,
                      labels, mode):
        return self._decoder(
            initial_state=initial_state,
            decoding_strategy=self._hparams.decoding_strategy_train,
            inputs=self.embed_target(features, labels, mode),
            sequence_length=labels['target_length'],
            embedding=self._tgt_embedder.embedding,
            mode=mode)

    def _decode_infer(self, initial_state, encoder_results, features,
                      labels, mode):
        start_token = self._tgt_vocab.bos_token_id
        start_tokens = tf.ones_like(features['source_length']) * start_token

        max_l = self._decoder.hparams.max_decoding_length_infer

        if self._hparams.beam_search_width > 1:
            outputs, final_state = beam_search_decode(
                decoder_or_cell=self._decoder,
                embedding=self._tgt_embedder.embedding,
                start_tokens=start_tokens,
                end_token=self._tgt_vocab.eos_token_id,
                beam_width=self._hparams.beam_search_width,
                initial_state=initial_state,
                max_decoding_length=max_l)
            sequence_length = final_state.lengths
        else:
            return self._decoder(
                initial_state=initial_state,
                decoding_strategy=self._hparams.decoding_strategy_infer,
                embedding=self._tgt_embedder.embedding,
                start_tokens=start_tokens,
                end_token=self._tgt_vocab.eos_token_id,
                mode=mode)

    def decode(self, encoder_results, features, labels, mode):
        """Decodes.
        """
        initial_state = self._connect(encoder_results, features, labels, mode)

        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            return self._decode_train(
                initial_state, encoder_results, features, labels, mode)
        else:
            return self._decode_infer(
                initial_state, encoder_results, features, labels, mode)

    def get_loss(self, decoder_results,)

    def _build(self, features, labels, params, mode, config=None):
        self._build_embedders()
        self._build_encoder()
        self._build_decoder()
        self._build_connector()

        encoder_results = self.encode(features, labels, mode)
        decoder_results = self.decode(encoder_results, features, labels, mode)
