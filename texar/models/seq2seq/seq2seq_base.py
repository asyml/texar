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
Base class for seq2seq models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from texar.models.model_base import ModelBase
from texar.losses.mle_losses import sequence_sparse_softmax_cross_entropy
from texar.data.data.paired_text_data import PairedTextData
from texar.core.optimization import get_train_op
from texar import HParams
from texar.utils import utils
from texar.utils.variables import collect_trainable_variables

# pylint: disable=too-many-instance-attributes, unused-argument,
# pylint: disable=too-many-arguments, no-self-use

__all__ = [
    "Seq2seqBase"
]


class Seq2seqBase(ModelBase):
    """Base class inherited by all seq2seq model classes.

    .. document private functions
    .. automethod:: _build
    """

    def __init__(self, data_hparams, hparams=None):
        ModelBase.__init__(self, hparams)

        self._data_hparams = HParams(data_hparams,
                                     PairedTextData.default_hparams())

        self._src_vocab = None
        self._tgt_vocab = None
        self._src_embedder = None
        self._tgt_embedder = None
        self._connector = None
        self._encoder = None
        self._decoder = None

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                "source_embedder": "WordEmbedder",
                "source_embedder_hparams": {},
                "target_embedder": "WordEmbedder",
                "target_embedder_hparams": {},
                "embedder_share": True,
                "embedder_hparams_share": True,
                "encoder": "UnidirectionalRNNEncoder",
                "encoder_hparams": {},
                "decoder": "BasicRNNDecoder",
                "decoder_hparams": {},
                "decoding_strategy_train": "train_greedy",
                "decoding_strategy_infer": "infer_greedy",
                "beam_search_width": 0,
                "connector": "MLPTransformConnector",
                "connector_hparams": {},
                "optimization": {},
                "name": "seq2seq",
            }

        Here:

        "source_embedder" : str or class or instance
            Word embedder for source text. Can be a class, its name or module
            path, or a class instance.

        "source_embedder_hparams" : dict
            Hyperparameters for constructing the source embedder. E.g.,
            See :meth:`~texar.modules.WordEmbedder.default_hparams` for
            hyperparameters of :class:`~texar.modules.WordEmbedder`. Ignored
            if "source_embedder" is an instance.

        "target_embedder", "target_embedder_hparams" :
            Same as "source_embedder" and "source_embedder_hparams" but for
            target text embedder.

        "embedder_share" : bool
            Whether to share the source and target embedder. If `True`,
            source embedder will be used to embed target text.

        "embedder_hparams_share" : bool
            Whether to share the embedder configurations. If `True`,
            target embedder will be created with "source_embedder_hparams".
            But the two embedders have different set of trainable variables.

        "encoder", "encoder_hparams" :
            Same as "source_embedder" and "source_embedder_hparams" but for
            encoder.

        "decoder", "decoder_hparams" :
            Same as "source_embedder" and "source_embedder_hparams" but for
            decoder.

        "decoding_strategy_train" : str
            The decoding strategy in training mode. See
            :meth:`~texar.modules.RNNDecoderBase._build` for details.

        "decoding_strategy_infer" : str
            The decoding strategy in eval/inference mode.

        "beam_search_width" : int
            Beam width. If > 1, beam search is used in eval/inference mode.

        "connector", "connector_hparams" :
            The connector class and hyperparameters. A connector transforms
            an encoder final state to a decoder initial state.

        "optimization" : dict
            Hyperparameters of optimizating the model. See
            :func:`~texar.core.default_optimization_hparams` for details.

        "name" : str
            Name of the model.
        """
        hparams = ModelBase.default_hparams()
        hparams.update({
            "name": "seq2seq",
            "source_embedder": "WordEmbedder",
            "source_embedder_hparams": {},
            "target_embedder": "WordEmbedder",
            "target_embedder_hparams": {},
            "embedder_share": True,
            "embedder_hparams_share": True,
            "encoder": "UnidirectionalRNNEncoder",
            "encoder_hparams": {},
            "decoder": "BasicRNNDecoder",
            "decoder_hparams": {},
            "decoding_strategy_train": "train_greedy",
            "decoding_strategy_infer": "infer_greedy",
            "beam_search_width": 0,
            "connector": "MLPTransformConnector",
            "connector_hparams": {},
            "optimization": {}
        })
        return hparams

    def _build_vocab(self):
        self._src_vocab, self._tgt_vocab = PairedTextData.make_vocab(
            self._data_hparams.source_dataset,
            self._data_hparams.target_dataset)

    def _build_embedders(self):
        kwargs = {
            "vocab_size": self._src_vocab.size,
            "hparams": self._hparams.source_embedder_hparams.todict()
        }
        self._src_embedder = utils.check_or_get_instance(
            self._hparams.source_embedder, kwargs,
            ["texar.modules", "texar.custom"])

        if self._hparams.embedder_share:
            self._tgt_embedder = self._src_embedder
        else:
            kwargs = {
                "vocab_size": self._tgt_vocab.size,
            }
            if self._hparams.embedder_hparams_share:
                kwargs["hparams"] = \
                        self._hparams.source_embedder_hparams.todict()
            else:
                kwargs["hparams"] = \
                        self._hparams.target_embedder_hparams.todict()
            self._tgt_embedder = utils.check_or_get_instance(
                self._hparams.target_embedder, kwargs,
                ["texar.modules", "texar.custom"])

    def _build_encoder(self):
        kwargs = {
            "hparams": self._hparams.encoder_hparams.todict()
        }
        self._encoder = utils.check_or_get_instance(
            self._hparams.encoder, kwargs,
            ["texar.modules", "texar.custom"])

    def _build_decoder(self):
        raise NotImplementedError

    def _build_connector(self):
        kwargs = {
            "output_size": self._decoder.state_size,
            "hparams": self._hparams.connector_hparams.todict()
        }
        self._connector = utils.check_or_get_instance(
            self._hparams.connector, kwargs,
            ["texar.modules", "texar.custom"])

    def get_loss(self, decoder_results, features, labels):
        """Computes the training loss.
        """
        return sequence_sparse_softmax_cross_entropy(
            labels=labels['target_text_ids'][:, 1:],
            logits=decoder_results['outputs'].logits,
            sequence_length=decoder_results['sequence_length'])

    def _get_predictions(self, decoder_results, features, labels, loss=None):
        raise NotImplementedError

    def _get_train_op(self, loss):
        varlist = collect_trainable_variables(
            [self._src_embedder, self._tgt_embedder, self._encoder,
             self._connector, self._decoder])
        return get_train_op(
            loss, variables=varlist, hparams=self._hparams.optimization)

    def _get_eval_metric_ops(self, decoder_results, features, labels):
        return None

    def embed_source(self, features, labels, mode):
        """Embeds the inputs.
        """
        raise NotImplementedError

    def embed_target(self, features, labels, mode):
        """Embeds the target inputs. Used in training.
        """
        raise NotImplementedError

    def encode(self, features, labels, mode):
        """Encodes the inputs.
        """
        raise NotImplementedError

    def _connect(self, encoder_results, features, labels, mode):
        """Transforms encoder final state into decoder initial state.
        """
        raise NotImplementedError

    def decode(self, encoder_results, features, labels, mode):
        """Decodes.
        """
        raise NotImplementedError

    def _build(self, features, labels, params, mode, config=None):
        self._build_vocab()
        self._build_embedders()
        self._build_encoder()
        self._build_decoder()
        self._build_connector()

        encoder_results = self.encode(features, labels, mode)
        decoder_results = self.decode(encoder_results, features, labels, mode)

        loss, train_op, preds, eval_metric_ops = None, None, None, None
        if mode == tf.estimator.ModeKeys.PREDICT:
            preds = self._get_predictions(decoder_results, features, labels)
        else:
            loss = self.get_loss(decoder_results, features, labels)

            if mode == tf.estimator.ModeKeys.TRAIN:
                train_op = self._get_train_op(loss)
            if mode == tf.estimator.ModeKeys.EVAL:
                eval_metric_ops = self._get_eval_metric_ops(
                    decoder_results, features, labels)

            preds = self._get_predictions(decoder_results, features, labels,
                                          loss)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=preds,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops)

    def get_input_fn(self, mode, hparams=None): #pylint:disable=arguments-differ
        """Creates an input function `input_fn` that provides input data
        for the model in an :tf_main:`Estimator <estimator/Estimator>`.
        See, e.g., :tf_main:`tf.estimator.train_and_evaluate
        <estimator/train_and_evaluate>`.

        Args:
            mode: One of members in
                :tf_main:`tf.estimator.ModeKeys <estimator/ModeKeys>`.
            hparams: A `dict` or an :class:`~texar.HParams` instance
                containing the hyperparameters of
                :class:`~texar.data.PairedTextData`. See
                :meth:`~texar.data.PairedTextData.default_hparams` for the
                the structure and default values of the hyperparameters.

        Returns:
            An input function that returns a tuple `(features, labels)`
            when called. `features` contains data fields that are related
            to source text, and `labels` contains data fields related
            to target text. See :class:`~texar.data.PairedTextData` for
            all data fields.
        """
        def _input_fn():
            data = PairedTextData(hparams)

            iterator = data.dataset.make_initializable_iterator()
            tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS,
                                 iterator.initializer)

            batch = iterator.get_next()

            features, labels = {}, {}
            for key, value in batch.items():
                if key.startswith('source_'):
                    features[key] = value
                else:
                    labels[key] = value
            return features, labels

        return _input_fn
