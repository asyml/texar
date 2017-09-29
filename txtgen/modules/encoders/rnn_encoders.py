#
"""
Various RNN encoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.contrib.rnn import LSTMStateTuple

from txtgen.modules.encoders.encoder_base import EncoderBase
from txtgen.core import layers


class RNNEncoderBase(EncoderBase):
    """Base class for all RNN encoder classes.

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
    """

    def __init__(self,  # pylint: disable=too-many-arguments
                 cell=None,
                 embedding=None,
                 embedding_trainable=True,
                 vocab_size=None,
                 hparams=None):
        EncoderBase.__init__(self, hparams)

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
                "embedding": txtgen.core.layers.default_embedding_hparams(),

                # The name of the encoder.
                "name": "rnn_encoder"
            }
        """
        return {
            "rnn_cell": layers.default_rnn_cell_hparams(),
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

    def __init__(self,  # pylint: disable=too-many-arguments
                 cell=None,
                 embedding=None,
                 embedding_trainable=True,
                 vocab_size=None,
                 hparams=None):
        RNNEncoderBase.__init__(
            self, cell, embedding, embedding_trainable, vocab_size, hparams)

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


class HierarchicalForwardRNNEncoder(RNNEncoderBase):
    """One directional forward RNN encoder with 2 levels.

    Useful for encoding structured long sequences, e.g. paragraphs, dialogs,
    etc.

    Expect 3D tensor input [B, T, U]
    B: batch size T: the major seq len U:the minor seq len
    The minor encoder encodes the inputs along the 2 axis
    The major encoder encodes the inputs along the 1st axis

    the minor encoder supports various types: RNN, bi-RNN, CNN, CBOW etc.

    See :class:`~txtgen.modules.encoders.rnn_encoders.RNNEncoderBase` for the
    arguments, and :meth:`default_hparams` for the default hyperparameters.
    """

    def __init__(self,  # pylint: disable=too-many-arguments
                 cell=None,
                 embedding=None,
                 embedding_trainable=True,
                 vocab_size=None,
                 hparams=None):
        RNNEncoderBase.__init__(
            self, cell, embedding, embedding_trainable, vocab_size, hparams)

        if self._hparams.minor_type == 'rnn':
            # Make minor rnn cell
            with tf.variable_scope(self.variable_scope):  # pylint: disable=not-context-manager
                self._minor_cell = layers.get_rnn_cell(self._hparams.minor_cell)

        elif self._hparams.minor_type == 'bow':
            raise ValueError("BOW encoder is not yet supported")

        elif self._hparams.minor_type == 'cnn':
            raise ValueError("CNN encoder is not yet supported")

        else:
            raise ValueError("Unknown minor type {}"
                             .format(self._hparams.minor_type))

    #TODO(zhiting): docs for hparams `minor_type` and `minor_cell`.
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
                "embedding": txtgen.core.layers.default_embedding_hparams(),

                # The name of the encoder
                "name": "hierarchical_forward_rnn_encoder"
            }
        """
        hparams = {
            "minor_type": "rnn",
            "minor_cell": layers.default_rnn_cell_hparams()
        }
        hparams.update(RNNEncoderBase.default_hparams())
        hparams["name"] = "hierarchical_forward_rnn_encoder"
        return hparams

    def _build(self, inputs, **kwargs):
        """Encodes the inputs.

        Args:
            inputs: 3D tensor input [B, T, U]
            **kwargs: Optional keyword arguments of `tensorflow.nn.dynamic_rnn`,
                such as `sequence_length`, `initial_state`, etc.

        Returns:
            Outputs and final state of the encoder.
        """

        major_len = array_ops.shape(inputs)[1]
        minor_len = array_ops.shape(inputs)[2]
        embed_dim = self.hparams.embedding.dim
        minor_cell_size = self.hparams.minor_cell.cell.kwargs.num_units

        embedded_inputs = tf.nn.embedding_lookup(self._embedding, inputs)

        # B x T x U x E - > (BxT) x U x E
        flat_embedded_inputs = tf.reshape(embedded_inputs,
                                          [-1, minor_len, embed_dim])

        # pylint: disable=not-context-manager
        with tf.variable_scope("minor_rnn"):
            minor_outs, minor_states = tf.nn.dynamic_rnn(
                cell=self._minor_cell,
                inputs=flat_embedded_inputs,
                dtype=tf.float32)

            if type(minor_states) is LSTMStateTuple:
                minor_states = minor_states.h

        # B x T x minor_size
        major_inputs = tf.reshape(minor_states, [-1, major_len,
                                                 minor_cell_size])

        with tf.variable_scope("major_rnn"):
            if ('dtype' not in kwargs) and ('initial_state' not in kwargs):
                results = tf.nn.dynamic_rnn(
                    cell=self._cell,
                    inputs=major_inputs,
                    dtype=tf.float32,
                    **kwargs)
            else:
                results = tf.nn.dynamic_rnn(
                    cell=self._cell,
                    inputs=major_inputs,
                    **kwargs)

        self._add_internal_trainable_variables()
        # Add trainable variables of `self._cell` which may be constructed
        # externally
        self._add_trainable_variable(self._cell.trainable_variables)
        self._built = True

        return results
