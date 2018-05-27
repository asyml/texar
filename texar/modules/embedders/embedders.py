#
"""
Various embedders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from texar.module_base import ModuleBase
from texar.modules.embedders import embedder_utils

__all__ = [
    #"EmbedderBase",
    "WordEmbedder"
]

#TODO(zhiting): add soft-embedder, position-embedder, embedder combiner

#class EmbedderBase(ModuleBase):
#    """TODO
#    """

class WordEmbedder(ModuleBase):
    """Simple word embedder that maps indexes into embeddings via lookup.

    Either :attr:`init_value` or :attr:`vocab_size` is required. If both are
    given, :attr:`init_value.shape[0]` must equal :attr:`vocab_size`.

    Args:
        init_value (optional): A `Tensor` or numpy array that contains the
            initial value of embeddings. It is typically of shape
            `[vocab_size, embedding dim]`

            If `None`, embedding is initialized as specified in
            :attr:`hparams["initializer"]`. Otherwise, the
            :attr:`"initializer"` and :attr:`"dim"`
            hyperparameters in :attr:`hparams` are ignored.
        vocab_size (int, optional): The vocabulary size. Required if
            :attr:`init_value` is not given.
        hparams (dict, optional): Embedder hyperparameters. If it is not
            specified, the default hyperparameter setting is used. See
            :attr:`default_hparams` for the sturcture and default values.
        mode (optional): Similar to :func:`~texar.core.layers.get_rnn_cell`.
    """

    def __init__(self, init_value=None, vocab_size=None, hparams=None,
                 mode=None):
        ModuleBase.__init__(self, hparams)

        if init_value is None and vocab_size is None:
            raise ValueError(
                "Either `init_value` or `vocab_size` is required.")

        self._vocab_size = vocab_size
        self._embedding = embedder_utils.get_embedding(
            self._hparams, init_value, self._vocab_size,
            self.variable_scope, mode)
        if self._hparams.trainable:
            self._add_trainable_variable(self._embedding)

        if self._vocab_size is None:
            self._vocab_size = self._embedding.get_shape().as_list()[0]

        self._dim = self._embedding.get_shape().as_list()[1:]
        if len(self._dim) == 1:
            self._dim = self._dim[0]

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
                    "dropout": {
                        "keep_prob": 1.0,
                    },
                    "trainable": True,
                }

            See :func:`~texar.modules.default_embedding_hparams` for more
            details.
        """
        hparams = embedder_utils.default_embedding_hparams()
        hparams["name"] = "word_embedder"
        return hparams

    def _build(self, inputs, **kwargs):
        """Embeds inputs with look-up.

        Args:
            inputs (Tensor): A `Tensor` with type `int32` or `int64`
                containing the ids to be looked up.
            kwargs: Additional keyword arguments for
                :tf_main:`tf.nn.embedding_lookup <nn/embedding_lookup>` besides
                :attr:`params` and :attr:`ids`.

        Returns:
            A `Tensor` of shape `shape(inputs) + embedding dimension`.
        """
        outputs = tf.nn.embedding_lookup(self._embedding, inputs, **kwargs)
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

