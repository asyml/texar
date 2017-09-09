#
"""
Various RNN encoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from txtgen.modules.encoders.encoder_base import EncoderBase
from txtgen.core.layers import get_rnn_cell
from txtgen.core.layers import default_rnn_cell_hparams
from txtgen.core import utils


class ForwardRNNEncoder(EncoderBase):
    """One directional forward RNN encoder.

    Args:
        cell: (RNNCell, optional) If it is not specified,
            a cell is created as specified in :attr:`hparams["rnn_cell"]`.
        embedding (optional): A `Variable` or a 2D `Tensor` (or `numpy array`)
            of shape `[vocab_size, embedding_dim]` that contains the token
            embeddings.

            If a `Variable`, it is directly used in encoding, and
            the hyperparameters in :attr:`hparams["embedding"]` is ignored.

            If a `Tensor` or `numpy array`, a new `Variable` is created taking
            :attr:`embedding` as initial value. The :attr:`"initializer"` and
            :attr:`"dim"` hyperparameters in :attr:`hparams["embedding"]` are
            ignored.

            If not given, a new `Variable` is created as specified in
            :attr:`hparams["embedding"]`.
        embedding_trainable (bool): If `True` (default), the encoder
            will update the embeddings during training. If `False`, the
            embeddings are not updated in the encoder, but might be updated
            elsewhere if they are created externally and used in other
            modules.
        vocab_size (int, optional): The vocabulary size. Required if
            :attr:`embedding` is not provided.
        hparams (dict, optional): Encoder hyperparameters. If it is not
            specified, the default hyperparameter setting is used. See
            :attr:`default_hparams` for the sturcture and default values.
        name (string): Name of the encoder.
    """

    def __init__(self,  # pylint: disable=too-many-arguments
                 cell=None,
                 embedding=None,
                 embedding_trainable=True,
                 vocab_size=None,
                 hparams=None,
                 name="forward_rnn_encoder"):
        EncoderBase.__init__(self, hparams, name)

        if cell is not None:
            self._cell = cell
        else:
            #a = self._hparams.rnn_cell
            #exit()
            self._cell = get_rnn_cell(self._hparams.rnn_cell)

        if embedding is None and vocab_size is None:
            raise ValueError("If `embedding` is not provided, `vocab_size` must"
                             " be specified.")
        self._embedding = embedding
        self._embedding_trainable = embedding_trainable
        self._vocab_size = vocab_size

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        The dictionary has the following structure and default values.

        See :class:`~txtgen.core.layers.default_rnn_cell_hparams` for the
        default rnn cell hyperparameters.

        .. code-block:: python

            {
                # A dictionary of rnn cell hyperparameters. See
                # `txtgen.core.layers.default_rnn_cell_hparams` for the
                # structure and default values. Ignored if `cell` is given
                # when constructing the encoder.
                "rnn_cell": default_rnn_cell_hparams,

                # A dictionary of token embedding hyperparameters for embedding
                # initialization. Ignored if `embedding` is given and is
                # a `Variable` when constructing the encoder.
                "embedding": {

                    # Embedding dimension. Ignored if `embedding` is given
                    # when constructing the encoder.
                    "dim": 100,

                    # Initializer of embedding values. Ignored if
                    # `embedding` is given when constructing the encoder.
                    "initializer": {
                        # A string. Name of the embedding variables.
                        "name": "embedding",

                        # A string. Name or full path to the initializer class.
                        # An initializer is a class inheriting from
                        # `tensorflow.Initializer`, which can be built-in
                        # classes in module `tensorflow`, or user-defined
                        # classes in `txtgen.custom`, or a full path like
                        # `my_module.MyInitializer`.
                        "type": "tensorflow.random_uniform_initializer",

                        # A dictionary of arguments for constructor of the
                        # initializer class. An initializer is created by
                        # calling `initialzier_class(**kwargs)` where
                        # `initializer_class` is specified in `type`.
                        "kwargs": {
                            "minval": -0.1,
                            "maxval": 0.1,
                            "seed": None
                        }
                    }
                }
            }
        """
        return {
            "rnn_cell": default_rnn_cell_hparams(),
            "embedding": { #TODO(zhiting): allow more hparams like regularizer
                "name": "embedding",
                "dim": 100,
                "initializer": {
                    "type": "tensorflow.random_uniform_initializer",
                    "kwargs": {
                        "minval": -0.1,
                        "maxval": 0.1,
                        "seed": None
                    }
                }
            }
        }

    def _build_embedding(self):
        if self._embedding is None:
            initializer = utils.get_instance(
                self._hparams.embedding.initializer.type,
                self._hparams.embedding.initializer.kwargs,
                ["txtgen.custom", "tensorflow"])
            self._embedding = tf.get_variable(
                name=self._hparams.embedding.name,
                shape=[self._vocab_size, self._hparams.embedding.dim],
                initializer=initializer,
                trainable=self._embedding_trainable)
        elif not isinstance(self._embedding, tf.Variable) and \
                not isinstance(self._embedding, tf.Constant):
            # self._embedding is a Tensor or numpy array
            self._embedding = tf.get_variable(
                name=self._hparams.embedding.name,
                initializer=self._embedding,
                trainable=self._embedding_trainable)


    def _build(self, inputs, **kwargs):
        """Encodes the inputs.

        Args:
            inputs: An integer Tensor containing Input sequences of
                token indexes.
            **kwargs: Optional keyword arguments of `tensorflow.nn.dynamic_rnn`,
                such as `sequence_length`, `initial_state`, etc.

        Returns:
            Outputs and final state of the encoder.
        """
        self._build_embedding()

        embedded_inputs = tf.nn.embedding_lookup(self._embedding, inputs)

        if ('dtype' not in kwargs) and ('initial_state' not in kwargs):
            results = tf.nn.dynamic_rnn(
                cell=self._cell,
                inputs=embedded_inputs,
                dtype=tf.float32,
                **kwargs)
        else:
            results = tf.nn.dynamic_rnn(
                cell=self._cell,
                inputs=embedded_inputs,
                **kwargs)

        self._add_internal_trainable_variables()
        # Add trainable variables of `self._cell` and `self._embedding` which
        # may be constructed externally
        self._add_trainable_variable(self._cell.trainable_variables)
        if self._embedding_trainable:
            self._add_trainable_variable(self._embedding)

        return results

