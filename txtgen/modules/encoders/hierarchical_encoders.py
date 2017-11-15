#
"""
Various encoders that encode data with hierarchical structure.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import array_ops # pylint: disable=E0611
from tensorflow.contrib.rnn import LSTMStateTuple

from txtgen.modules.encoders.encoder_base import EncoderBase
from txtgen.core import layers

#TODO(zhiting): this is incomplete
__all__ = [
    "HierarchicalEncoder"
]

class HierarchicalEncoder(EncoderBase):
    """One directional forward RNN encoder with 2 levels.

    Useful for encoding structured long sequences, e.g. paragraphs, dialogs,
    etc.

    Expect 3D tensor input [B, T, U]
    B: batch size T: the major seq len U:the minor seq len
    The major encoder encodes the inputs along the 1st axis
    The minor encoder encodes the inputs along the 2nd axis

    the minor encoder supports various types: RNN, bi-RNN, CNN, CBOW etc.

    See :class:`~txtgen.modules.encoders.rnn_encoders.RNNEncoderBase` for the
    arguments, and :meth:`default_hparams` for the default hyperparameters.
    """

    def __init__(self, cell=None,
                 embedding=None,
                 vocab_size=None,
                 hparams=None):
        EncoderBase.__init__(self, hparams)

        # Make embedding
        self._embedding = None
        if self._hparams.use_embedding:
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

        # Make RNN cell
        with tf.variable_scope(self.variable_scope):
            if cell is not None:
                self._major_cell = cell
            else:
                self._major_cell = layers.get_rnn_cell(self._hparams.major_cell)

        if self._hparams.minor_type == 'rnn':
            # Make minor rnn cell
            with tf.variable_scope(self.variable_scope):
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
            "use_embedding": True,
            "embedding": layers.default_embedding_hparams(),
            "minor_type": "rnn",
            "minor_cell": layers.default_rnn_cell_hparams(),
            "major_cell": layers.default_rnn_cell_hparams(),
        }
        hparams.update(EncoderBase.default_hparams())
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
                    cell=self._major_cell,
                    inputs=major_inputs,
                    dtype=tf.float32,
                    **kwargs)
            else:
                results = tf.nn.dynamic_rnn(
                    cell=self._major_cell,
                    inputs=major_inputs,
                    **kwargs)

        self._add_internal_trainable_variables()
        # Add trainable variables of `self._cell` which may be constructed
        # externally
        self._add_trainable_variable(
            layers.get_rnn_cell_trainable_variables(self._major_cell))
        self._built = True

        return results
