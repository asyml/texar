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
    given, there must be `init_value.shape[0]==vocab_size`.

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
        hparams (dict, optional): Embedder hyperparameters. Missing
            hyperparamerter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter sturcture and
            default values.

    See :meth:`_build` for the inputs and outputs of the embedder.

    Example:

        .. code-block:: python

            ids = tf.random_uniform(shape=[32, 10], maxval=10, dtype=tf.int64)
            soft_ids = tf.random_uniform(shape=[32, 10, 100])

            embedder = WordEmbedder(vocab_size=100, hparams={'dim': 256})
            ids_emb = embedder(ids=ids) # shape: [32, 10, 256]
            soft_ids_emb = embedder(soft_ids=soft_ids) # shape: [32, 10, 256]

        .. code-block:: python

            ## Use with Texar data module
            hparams={
                'dataset': {
                    'embedding_init': {'file': 'word2vec.txt'}
                    ...
                },
            }
            data = MonoTextData(data_params)
            iterator = DataIterator(data)
            batch = iterator.get_next()

            # Use data vocab size
            embedder_1 = WordEmbedder(vocab_size=data.vocab.size)
            emb_1 = embedder_1(batch['text_ids'])

            # Use pre-trained embedding
            embedder_2 = WordEmbedder(init_value=data.embedding_init_value)
            emb_2 = embedder_2(batch['text_ids'])


    .. document private functions
    .. automethod:: _build
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

        .. code-block:: python

            {
                "dim": 100,
                "dropout_rate": 0,
                "dropout_strategy": 'element',
                "trainable": True,
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
                "name": "word_embedder",
            }

        Here:

        "dim" : int or list
            Embedding dimension. Can be a list of integers to yield embeddings
            with dimensionality > 1.

            Ignored if :attr:`init_value` is given to the embedder constructor.

        "dropout_rate" : float
            The dropout rate between 0 and 1. E.g., `dropout_rate=0.1` would
            drop out 10% of the embedding. Set to 0 to disable dropout.

        "dropout_strategy" : str
            The dropout strategy. Can be one of the following

            - :attr:`"element"`: The regular strategy that drops individual \
            elements of embedding vectors.
            - :attr:`"item"`: Drops individual items (e.g., words) entirely. \
            E.g., for \
            the word sequence 'the simpler the better', the strategy can \
            yield '_ simpler the better', where the first `the` is dropped.
            - :attr:`"item_type"`: Drops item types (e.g., word types). \
            E.g., for the \
            above sequence, the strategy can yield '_ simpler _ better', \
            where the word type 'the' is dropped. The dropout will never \
            yield '_ simpler the better' as in the 'item' strategy.

        "trainable" : bool
            Whether the embedding is trainable.

        "initializer" : dict or None
            Hyperparameters of the initializer for embedding values. See
            :func:`~texar.core.get_initializer` for the details. Ignored if
            :attr:`init_value` is given to the embedder constructor.

        "regularizer" : dict
            Hyperparameters of the regularizer for embedding values. See
            :func:`~texar.core.get_regularizer` for the details.

        "name" : str
            Name of the embedding variable.
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
            soft_ids (optional): A tensor of weights (probabilities) used to
                mix the embedding vectors.
            mode (optional): A tensor taking value in
                :tf_main:`tf.estimator.ModeKeys <estimator/ModeKeys>`, including
                `TRAIN`, `EVAL`, and `PREDICT`. If `None`, dropout is
                controlled by :func:`texar.global_mode`.
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
        """The embedding tensor, of shape `[vocab_size] + dim`.
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
