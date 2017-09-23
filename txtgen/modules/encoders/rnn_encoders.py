#
"""
Various RNN encoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from txtgen.modules.encoders.encoder_base import EncoderBase
from txtgen.core import layers


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

        # Make rnn cell
        with tf.variable_scope(self.variable_scope): # pylint: disable=not-context-manager
            if cell is not None:
                self._cell = cell
            else:
                self._cell = layers.get_rnn_cell(self._hparams.rnn_cell)

        # Make embedding
        if embedding is None and vocab_size is None:
            raise ValueError("If `embedding` is not provided, `vocab_size` "
                             "must be specified.")
        if isinstance(embedding, tf.Variable):
            self._embedding = embedding
        else:
            self._embedding = layers.get_embedding(
                self._hparams.embedding, embedding, vocab_size,
                embedding_trainable, self.variable_scope)
        if embedding_trainable:
            self._add_trainable_variable(self._embedding)

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        The dictionary has the following structure and default values.

        See :meth:`~txtgen.core.layers.default_rnn_cell_hparams` for the
        default rnn cell hyperparameters, and
        :meth:`~txtgen.core.layers.default_embedding_hparams` for the default
        embedding hyperparameters.

        .. code-block:: python

            {
                # A dictionary of rnn cell hyperparameters. Ignored if `cell`
                # is given when constructing the encoder.
                "rnn_cell": txtgen.core.layers.default_rnn_cell_hparams(),

                # A dictionary of token embedding hyperparameters for embedding
                # initialization. Ignored if `embedding` is given and is
                # a tf.Variable when constructing the encoder. If `embedding`
                # is given and is a Tensor or numpy array, the "dim" and
                # "initializer" specs of embedding are ignored.
                "embedding": txtgen.core.layers.default_embedding_hparams()
            }
        """
        return {
            "rnn_cell": layers.default_rnn_cell_hparams(),
            "embedding": layers.default_embedding_hparams()
        }


    def _build(self, inputs, **kwargs):
        """Encodes the inputs.

        Args:
            inputs: An integer Tensor containing input sequences of
                token indexes.
            **kwargs: Optional keyword arguments of `tensorflow.nn.dynamic_rnn`,
                such as `sequence_length`, `initial_state`, etc.

        Returns:
            Outputs and final state of the encoder.
        """
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
        # Add trainable variables of `self._cell` which may be constructed
        # externally
        self._add_trainable_variable(self._cell.trainable_variables)
        self._built = True

        return results


    @property
    def embedding(self):
        """The embedding variable.
        """
        return self._embedding

    @property
    def cell(self):
        """The RNN cell.
        """
        return self._cell

    @property
    def state_size(self):
        """The state size of encoder cell.

        Same as :attr:`encoder.cell.state_size`.
        """
        return self.cell.state_size
