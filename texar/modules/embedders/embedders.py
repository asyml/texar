#
"""
Various embedders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from texar.modules.embedders.embedder_base import EmbedderBase
from texar.modules.embedders import embedder_utils
from texar.utils.mode import is_train_mode
from texar.utils.shapes import get_rank

__all__ = [
    "WordEmbedder"
]

class WordEmbedder(EmbedderBase):
    """Simple word embedder that maps indexes into embeddings. The indexes
    can be soft (e.g., distributions over vocabulary).

    Either :attr:`init_value` or :attr:`vocab_size` is required. If both are
    given, :attr:`init_value.shape[0]` must equal :attr:`vocab_size`.

    Args:
        init_value (optional): A `Tensor` or numpy array that contains the
            initial value of embeddings. It is typically of shape
            `[vocab_size] + embedding-dim`. Embedding can have dimensionality
            > 1.

            If `None`, embedding is initialized as specified in
            :attr:`hparams["initializer"]`. Otherwise, the
            :attr:`"initializer"` and :attr:`"dim"`
            hyperparameters in :attr:`hparams` are ignored.
        vocab_size (int, optional): The vocabulary size. Required if
            :attr:`init_value` is not given.
        hparams (dict, optional): Embedder hyperparameters. If it is not
            specified, the default hyperparameter setting is used. See
            :attr:`default_hparams` for the sturcture and default values.
    """

    def __init__(self, init_value=None, vocab_size=None, hparams=None):
        EmbedderBase.__init__(self, hparams=hparams)

        if init_value is None and vocab_size is None:
            raise ValueError(
                "Either `init_value` or `vocab_size` is required.")

        self._init_parameterized_embedding(init_value, vocab_size,
                                           self._hparams)

        self._vocab_size = vocab_size
        if vocab_size is None:
            self._vocab_size = self._num_embeds
        if self._vocab_size != self._num_embeds:
            raise ValueError(
                'vocab_size must equal to init_value.shape[0].'
                'Got %d and %d' % (self._vocab_size, self._num_embeds))

        self._built = True

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        Returns:
            A dictionary with the following structure and values.

            .. code-block:: python

                {
                    "name": "word_embedder",
                    "dim": 100,
                    "initializer": {
                        "type": "random_uniform_initializer",
                        "kwargs": {
                            "minval": -0.1,
                            "maxval": 0.1,
                            "seed": None
                        }
                    },
                    "regularizer": {
                        "type": "L1L2",
                        "kwargs": {
                            "l1": 0.,
                            "l2": 0.
                        }
                    },
                    "dropout_rate": 0,
                    "dropout_strategy": 'element',
                    "trainable": True,
                }

            See :func:`~texar.modules.default_embedding_hparams` for more
            details.
        """
        hparams = embedder_utils.default_embedding_hparams()
        hparams["name"] = "word_embedder"
        return hparams

    def _build(self, ids=None, soft_ids=None, mode=None, **kwargs):
        """Embeds (soft) ids.

        Either :attr:`ids` or :attr:`soft_ids` must be given, and they
        must not be given at the same time.

        Args:
            ids (optional): An integer tensor containing the ids to embed.
            soft_ids (optional): A Tensor of weights (probabilities) used to
                mix the embedding vectors.
            mode (optional): A tensor taking value in
                :tf_main:`tf.estimator.ModeKeys <estimator/ModeKeys>`, including
                `TRAIN`, `EVAL`, and `PREDICT`. If `None`, dropout will be
                controlled by :func:`texar.context.global_mode`.
            kwargs: Additional keyword arguments for
                :tf_main:`tf.nn.embedding_lookup <nn/embedding_lookup>` besides
                :attr:`params` and :attr:`ids`.

        Returns:
            If :attr:`ids` is given, returns a Tensor of shape
            `shape(ids) + embedding-dim`. For example,
            if `shape(ids) = [batch_size, max_time]`
            and `shape(embedding) = [vocab_size, emb_dim]`, then the return
            tensor has shape `[batch_size, max_time, emb_dim]`.

            If :attr:`soft_ids` is given, returns a Tensor of shape
            `shape(soft_ids)[:-1] + embdding-dim`. For example,
            if `shape(soft_ids) = [batch_size, max_time, vocab_size]`
            and `shape(embedding) = [vocab_size, emb_dim]`, then the return
            tensor has shape `[batch_size, max_time, emb_dim]`.
        """
        if ids is not None:
            if soft_ids is not None:
                raise ValueError(
                    'Must not specify `ids` and `soft_ids` at the same time.')
            ids_rank = get_rank(ids)
        elif soft_ids is not None:
            ids_rank = get_rank(soft_ids) - 1
        else:
            raise ValueError('Either `ids` or `soft_ids` must be given.')

        embedding = self._embedding

        is_training = is_train_mode(mode)
        if self._hparams.dropout_strategy == 'item_type':
            dropout_layer = self._get_dropout_layer(self._hparams)
            if dropout_layer:
                embedding = dropout_layer.apply(inputs=embedding,
                                                training=is_training)

        if ids is not None:
            outputs = tf.nn.embedding_lookup(embedding, ids, **kwargs)
        else:
            outputs = embedder_utils.soft_embedding_lookup(embedding, soft_ids)

        if self._hparams.dropout_strategy != 'item_type':
            dropout_layer = self._get_dropout_layer(
                self._hparams, ids_rank=ids_rank, dropout_input=outputs)
            if dropout_layer:
                outputs = dropout_layer.apply(
                    inputs=outputs, training=is_training)

        return outputs

    @property
    def embedding(self):
        """The embedding tensor.
        """
        return self._embedding

    @property
    def dim(self):
        """The embedding dimension.
        """
        return self._dim

    @property
    def vocab_size(self):
        """The vocabulary size.
        """
        return self._vocab_size

