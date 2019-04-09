# Copyright 2018 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The basic seq2seq model without attention.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from texar.models.seq2seq.seq2seq_base import Seq2seqBase
from texar.modules.decoders.beam_search_decode import beam_search_decode
from texar.utils import utils
from texar.utils.shapes import get_batch_size

# pylint: disable=protected-access, too-many-arguments, unused-argument

__all__ = [
    "BasicSeq2seq"
]


class BasicSeq2seq(Seq2seqBase):
    """The basic seq2seq model (without attention).

    Example:

        .. code-block:: python

            model = BasicSeq2seq(data_hparams, model_hparams)
            exor = tx.run.Executor(
                model=model,
                data_hparams=data_hparams,
                config=run_config)
            exor.train_and_evaluate(
                max_train_steps=10000,
                eval_steps=100)

    .. document private functions
    .. automethod:: _build
    """

    def __init__(self, data_hparams, hparams=None):
        Seq2seqBase.__init__(self, data_hparams, hparams=hparams)

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        Same as :meth:`~texar.models.Seq2seqBase.default_hparams` of
        :class:`~texar.models.Seq2seqBase`.
        """
        hparams = Seq2seqBase.default_hparams()
        hparams.update({
            "name": "basic_seq2seq"
        })
        return hparams

    def _build_decoder(self):
        kwargs = {
            "vocab_size": self._tgt_vocab.size,
            "hparams": self._hparams.decoder_hparams.todict()
        }
        self._decoder = utils.check_or_get_instance(
            self._hparams.decoder, kwargs,
            ["texar.modules", "texar.custom"])

    def _get_predictions(self, decoder_results, features, labels, loss=None):
        preds = {}

        preds.update(features)

        if labels is not None:
            preds.update(labels)

        preds.update(utils.flatten_dict({'decode': decoder_results}))
        preds['decode.outputs.sample'] = self._tgt_vocab.map_ids_to_tokens(
            preds['decode.outputs.sample_id'])

        if loss is not None:
            preds['loss'] = loss

        return preds

    def embed_source(self, features, labels, mode):
        """Embeds the inputs.
        """
        return self._src_embedder(ids=features["source_text_ids"], mode=mode)

    def embed_target(self, features, labels, mode):
        """Embeds the target inputs. Used in training.
        """
        return self._tgt_embedder(ids=labels["target_text_ids"], mode=mode)

    def encode(self, features, labels, mode):
        """Encodes the inputs.
        """
        embedded_source = self.embed_source(features, labels, mode)

        outputs, final_state = self._encoder(
            embedded_source,
            sequence_length=features["source_length"],
            mode=mode)

        return {'outputs': outputs, 'final_state': final_state}

    def _connect(self, encoder_results, features, labels, mode):
        """Transforms encoder final state into decoder initial state.
        """
        enc_state = encoder_results["final_state"]
        possible_kwargs = {
            "inputs": enc_state,
            "batch_size": get_batch_size(enc_state)
        }
        outputs = utils.call_function_with_redundant_kwargs(
            self._connector._build, possible_kwargs)
        return outputs

    def _decode_train(self, initial_state, encoder_results, features,
                      labels, mode):
        return self._decoder(
            initial_state=initial_state,
            decoding_strategy=self._hparams.decoding_strategy_train,
            inputs=self.embed_target(features, labels, mode),
            sequence_length=labels['target_length']-1,
            mode=mode)

    def _decode_infer(self, initial_state, encoder_results, features,
                      labels, mode):
        start_token = self._tgt_vocab.bos_token_id
        start_tokens = tf.ones_like(features['source_length']) * start_token

        max_l = self._decoder.hparams.max_decoding_length_infer

        if self._hparams.beam_search_width > 1:
            return beam_search_decode(
                decoder_or_cell=self._decoder,
                embedding=self._tgt_embedder.embedding,
                start_tokens=start_tokens,
                end_token=self._tgt_vocab.eos_token_id,
                beam_width=self._hparams.beam_search_width,
                initial_state=initial_state,
                max_decoding_length=max_l)
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

        if mode == tf.estimator.ModeKeys.PREDICT:
            outputs, final_state, sequence_length = self._decode_infer(
                initial_state, encoder_results, features, labels, mode)
        else:
            outputs, final_state, sequence_length = self._decode_train(
                initial_state, encoder_results, features, labels, mode)

        return {'outputs': outputs,
                'final_state': final_state,
                'sequence_length': sequence_length}

