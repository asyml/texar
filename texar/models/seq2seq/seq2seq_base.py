#
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
from texar.utils import utils
from texar.utils.variables import collect_trainable_variables

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

    def get_loss(self, decoder_results, features, labels):
        """Computes the training loss.
        """
        return sequence_sparse_softmax_cross_entropy(
            labels=labels['target_text_ids'][:, 1:],
            logist=decoder_results['outputs'].logits,
            sequence_length=decoder_results['outputs']['sequence_length'])

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

    def _build(self, features, labels, params, mode, config=None):
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

    def get_input_fn(self, mode, hparams=None):
        """Creates an input function `input_fn` that provides input data
        for the model in an :tf_main:`Estimator <estimator/Estimator>`.
        See, e.g., :tf_main:`tf.estimator.train_and_evaluate
        <estimator/train_and_evaluate>`.

        Args:
            mode: One of members in
                :tf_main:`tf.estimator.ModeKeys <estimator/ModeKeys>`.
            hparams: A `dict` or an :class:`~texar.hparams.HParams` instance
                containing the hyperparameters of
                :class:`~texar.data.PairedTextData`. See
                :meth:`~texar.data.PairedTextData.default_hparams` for the
                the structure and default values of hyperparameters.

        Returns:
            An input function that returns a tuple `(features, labels)`
            when called.
        """
        def _input_fn():
            data = PairedTextData(hparams)
            iterator = data.dataset.make_one_shot_iterator()
            batch = iterator.get_next()
            features, labels = {}, {}
            for key, value in batch:
                if key.startswith('source_'):
                    features[key] = value
                else:
                    labels[key] = value
            return features, labels

        return _input_fn

