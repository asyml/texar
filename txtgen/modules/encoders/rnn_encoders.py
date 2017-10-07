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

# pylint: disable=not-context-manager, too-many-arguments

class RNNEncoderBase(EncoderBase):
    """Base class for all RNN encoder classes.

    Args:
        cell: (RNNCell, optional) If it is not specified,
            a cell is created as specified in :attr:`hparams["rnn_cell"]`.
        embedding (optional): A `Variable` or a 2D `Tensor` (or `array`)
            of shape `[vocab_size, embedding_dim]` that contains the token
            embeddings. Ignore if :attr:`hparams["embedding_enabled"]`
            is `False`. If :attr:`hparams["embedding_enabled"]` is `True`:

            If a `Variable`, it is directly used in encoding, and
            hyperparameters in :attr:`hparams["embedding"]` are ignored.

            If a `Tensor` or `array`, it is used to initialize the token
            embedding variable. The :attr:`"initializer"` and :attr:`"dim"`
            hyperparameters in :attr:`hparams["embedding"]` are ignored.

            If not given, a new `Variable` is created as specified in
            :attr:`hparams["embedding"]`.
        vocab_size (int, optional): The vocabulary size. Required if
            :attr:`hparams["embedding_enabled"]` is `True` (default) and
            :attr:`embedding` is not provided.
        hparams (dict, optional): Encoder hyperparameters. If it is not
            specified, the default hyperparameter setting is used. See
            :attr:`default_hparams` for the sturcture and default values.
    """

    def __init__(self,
                 cell=None,
                 embedding=None,
                 vocab_size=None,
                 hparams=None):
        EncoderBase.__init__(self, hparams)

        # Make RNN cell
        with tf.variable_scope(self.variable_scope):
            if cell is not None:
                self._cell = cell
            else:
                self._cell = layers.get_rnn_cell(self._hparams.rnn_cell)

        # Make embedding
        self._embedding = None
        if self._hparams.embedding_enabled:
            if embedding is None and vocab_size is None:
                raise ValueError(
                    "`vocab_szie` is required if embedding is enabled and "
                    "`embedding` is not provided")
            if isinstance(embedding, tf.Variable):
                self._embedding = embedding
            else:
                self._embedding = layers.get_embedding(
                    self._hparams.embedding, embedding, vocab_size,
                    self.variable_scope)
            if self._hparams.embedding.trainable:
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

                # (bool) Wether embedding is used in the encoder. If `True`
                # (default), input to the encoder should contain integer
                # indexes and will be used to look up the embedding vectors.
                # If `False`, the input is directly fed into the RNN to encode.
                "embedding_enabled": True,

                # A dictionary of token embedding hyperparameters for embedding
                # initialization.
                #
                # Ignored if "embedding_enabled" is `False`, or a tf.Variable
                # is given to `embedding` in the encoder constructor. Note that
                # in the second case, the embedding variable might be updated
                # outside the encoder even if "embedding.trainable" is set to
                # `False` and not updated by the encoder.
                #
                # If a Tensor or array is given to `embedding` in the
                # constructor, "dim" and "initializer" in the configuration
                # are ignored.
                "embedding": txtgen.core.layers.default_embedding_hparams(),

                # Name of the encoder.
                "name": "rnn_encoder"
            }
        """
        return {
            "rnn_cell": layers.default_rnn_cell_hparams(),
            "embedding_enabled": True,
            "embedding": layers.default_embedding_hparams(),
            "name": "rnn_encoder"
        }

    def _build(self, inputs, *args, **kwargs):
        """Encodes the inputs.

        Args:
          inputs: Inputs to the encoder.
          *args: Other arguments.
          **kwargs: Keyword arguments.

        Returns:
          Encoding results.
        """
        raise NotImplementedError

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


class ForwardRNNEncoder(RNNEncoderBase):
    """One directional forward RNN encoder.

    See :class:`~txtgen.modules.encoders.rnn_encoders.RNNEncoderBase` for the
    arguments, and :meth:`default_hparams` for the default hyperparameters.
    """

    def __init__(self,
                 cell=None,
                 embedding=None,
                 vocab_size=None,
                 hparams=None):
        RNNEncoderBase.__init__(self, cell, embedding, vocab_size, hparams)

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        The dictionary have the same structure as in
        :meth:`RNNEncoderBase.default_hparams`, except the "name" is default to
        "forward_rnn_encoder".
        """
        hparams = RNNEncoderBase.default_hparams()
        hparams["name"] = "forward_rnn_encoder"
        return hparams

    def _build(self, inputs, **kwargs):
        """Encodes the inputs.

        Args:
            inputs: If embedding is enabled, this must be a 2D Tensor of shape
                `[batch_size, max_time]` containing input sequences of integer
                token indexes. Otherwise, this must be a 3D Tensor of shape
                `[batch_size, max_time, dim]`. The first two dimensions
                `batch_size` and `max_time` may be exchanged if
                `time_major=True` is specified.
            **kwargs: Optional keyword arguments of `tensorflow.nn.dynamic_rnn`,
                such as `sequence_length`, `initial_state`, etc.

        Returns:
            Outputs and final state of the encoder.
        """
        if self._embedding is not None:
            embedded_inputs = tf.nn.embedding_lookup(self._embedding, inputs)
        else:
            embedded_inputs = inputs

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


