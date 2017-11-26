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

__all__ = [
    "RNNEncoderBase", "UnidirectionalRNNEncoder"
]

class RNNEncoderBase(EncoderBase):
    """Base class for all RNN encoder classes.

    Args:
        embedding (optional): A `Variable` or a 2D `Tensor` (or `array`)
            of shape `[vocab_size, embedding_dim]` that contains the token
            embeddings. Ignore if :attr:`hparams["use_embedding"]`
            is `False`. If :attr:`hparams["use_embedding"]` is `True`:

            If a `Variable`, it is directly used in encoding, and
            hyperparameters in :attr:`hparams["embedding"]` are ignored.

            If a `Tensor` or `array`, it is used to initialize the token
            embedding variable. The :attr:`"initializer"` and :attr:`"dim"`
            hyperparameters in :attr:`hparams["embedding"]` are ignored.

            If not given, a new `Variable` is created as specified in
            :attr:`hparams["embedding"]`.
        vocab_size (int, optional): The vocabulary size. Required if
            :attr:`hparams["use_embedding"]` is `True` (default) and
            :attr:`embedding` is not provided.
        hparams (dict, optional): Encoder hyperparameters. If it is not
            specified, the default hyperparameter setting is used. See
            :attr:`default_hparams` for the sturcture and default values.
    """

    def __init__(self,
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

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        Returns:
            .. code-block:: python

                {
                    "use_embedding": True,
                    "embedding": txtgen.core.layers.default_embedding_hparams(),
                    "name": "unidirectional_rnn_encoder"
                }

            Here:

            "use_embedding" : bool
                Whether token embedding is used.

                If `True` (default), input to the encoder should contain integer
                indexes and will be used to look up the embedding vectors.
                If `False`, the input is directly fed into the RNN to encode.

            "embedding" : dict
                A dictionary of token embedding hyperparameters for
                embedding initialization.

                Ignored if :attr:`"use_embedding"` is `False`, or if
                :attr:`embedding` is given and is `Variable` when
                constructing the decoder.

                If :attr:`embedding` is given and is a Tensor or array, the
                "dim" and "initializer" specs in the embedding hyperparameterrs
                are ignored.

                The default value is defined in
                :meth:`~txtgen.core.layers.default_embedding_hparams`.

            "name" : str
                Name of the encoder.
        """
        return {
            "use_embedding": True,
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


class UnidirectionalRNNEncoder(RNNEncoderBase):
    """One directional RNN encoder.

    Args:
        cell: (RNNCell, optional) If it is not specified,
            a cell is created as specified in :attr:`hparams["rnn_cell"]`.
        embedding (optional): A `Variable` or a 2D `Tensor` (or `array`)
            of shape `[vocab_size, embedding_dim]` that contains the token
            embeddings. Ignore if :attr:`hparams["use_embedding"]`
            is `False`. If :attr:`hparams["use_embedding"]` is `True`:

            If a `Variable`, it is directly used in encoding, and
            hyperparameters in :attr:`hparams["embedding"]` are ignored.

            If a `Tensor` or `array`, it is used to initialize the token
            embedding variable. The :attr:`"initializer"` and :attr:`"dim"`
            hyperparameters in :attr:`hparams["embedding"]` are ignored.

            If not given, a new `Variable` is created as specified in
            :attr:`hparams["embedding"]`.
        vocab_size (int, optional): The vocabulary size. Required if
            :attr:`hparams["use_embedding"]` is `True` (default) and
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
        RNNEncoderBase.__init__(self, embedding, vocab_size, hparams)

        # Make RNN cell
        with tf.variable_scope(self.variable_scope):
            if cell is not None:
                self._cell = cell
            else:
                self._cell = layers.get_rnn_cell(self._hparams.rnn_cell)

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        Returns:
            .. code-block:: python

                {
                    "rnn_cell": default_rnn_cell_hparams(),
                    "use_embedding": True,
                    "embedding": default_embedding_hparams(),
                    "name": "unidirectional_rnn_encoder"
                }

            Here:

            "rnn_cell" : dict
                A dictionary of RNN cell hyperparameters. Ignored if
                :attr:`cell` is given when constructing the encoder.

                The default value is defined in
                :meth:`~txtgen.core.layers.default_rnn_cell_hparams`.

            "use_embedding" : bool
                Whether token embedding is used.

                If `True` (default), input to the encoder should contain integer
                indexes and will be used to look up the embedding vectors.
                If `False`, the input is directly fed into the RNN to encode.

            "embedding" : dict
                A dictionary of token embedding hyperparameters for
                embedding initialization.

                Ignored if :attr:`"use_embedding"` is `False`, or if
                :attr:`embedding` is given and is `Variable` when
                constructing the decoder.

                If :attr:`embedding` is given and is a Tensor or array, the
                "dim" and "initializer" specs in the embedding hyperparameterrs
                are ignored.

                The default value is defined in
                :meth:`~txtgen.core.layers.default_embedding_hparams`.

            "name" : str
                Name of the encoder
        """
        hparams = RNNEncoderBase.default_hparams()
        hparams["rnn_cell"] = layers.default_rnn_cell_hparams()
        hparams["name"] = "unidirectional_rnn_encoder"
        return hparams

    def _build(self, inputs, sequence_length=None, initial_state=None,
               **kwargs):
        """Encodes the inputs.

        Args:
            inputs: If embedding is enabled, this must be a 2D Tensor of shape
                `[batch_size, max_time]` containing input sequences of integer
                token indexes. Otherwise, this must be a 3D Tensor of shape
                `[batch_size, max_time, dim]`. The first two dimensions
                `batch_size` and `max_time` may be exchanged if
                `time_major=True` is specified.
            sequence_length (int list or 1D Tensor, optional): Sequence lengths
                of the batch inputs. Used to copy-through state and zero-out
                outputs when past a batch element's sequence length.
            initial_state (optional): Initial state of the RNN.
            **kwargs: Optional keyword arguments of
                :tf_main:`tf.nn.dynamic_rnn <nn/dynamic_rnn>`,
                such as `time_major`, `dtype`, etc.

        Returns:
            Outputs and final state of the encoder.
        """
        #TODO(zhiting): add docs of 'Returns'
        if self._embedding is not None:
            embedded_inputs = tf.nn.embedding_lookup(self._embedding, inputs)
        else:
            embedded_inputs = inputs

        if ('dtype' not in kwargs) and (initial_state is None):
            outputs, state = tf.nn.dynamic_rnn(
                cell=self._cell,
                inputs=embedded_inputs,
                sequence_length=sequence_length,
                initial_state=initial_state,
                dtype=tf.float32,
                **kwargs)
        else:
            outputs, state = tf.nn.dynamic_rnn(
                cell=self._cell,
                inputs=embedded_inputs,
                sequence_length=sequence_length,
                initial_state=initial_state,
                **kwargs)

        if not self._built:
            self._add_internal_trainable_variables()
            # Add trainable variables of `self._cell` which may be constructed
            # externally.
            self._add_trainable_variable(
                layers.get_rnn_cell_trainable_variables(self._cell))
            self._built = True

        return outputs, state

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


class BidirectionalRNNEncoder(RNNEncoderBase):
    """Bidirectional forward-backward RNN encoder.

    Args:
        cell: (RNNCell, optional) If it is not specified,
            a cell is created as specified in :attr:`hparams["rnn_cell"]`.
        embedding (optional): A `Variable` or a 2D `Tensor` (or `array`)
            of shape `[vocab_size, embedding_dim]` that contains the token
            embeddings. Ignore if :attr:`hparams["use_embedding"]`
            is `False`. If :attr:`hparams["use_embedding"]` is `True`:

            If a `Variable`, it is directly used in encoding, and
            hyperparameters in :attr:`hparams["embedding"]` are ignored.

            If a `Tensor` or `array`, it is used to initialize the token
            embedding variable. The :attr:`"initializer"` and :attr:`"dim"`
            hyperparameters in :attr:`hparams["embedding"]` are ignored.

            If not given, a new `Variable` is created as specified in
            :attr:`hparams["embedding"]`.
        vocab_size (int, optional): The vocabulary size. Required if
            :attr:`hparams["use_embedding"]` is `True` (default) and
            :attr:`embedding` is not provided.
        hparams (dict, optional): Encoder hyperparameters. If it is not
            specified, the default hyperparameter setting is used. See
            :attr:`default_hparams` for the sturcture and default values.
    """

    def __init__(self,
                 cell_fw=None,
                 cell_bw=None,
                 embedding=None,
                 vocab_size=None,
                 hparams=None):
        RNNEncoderBase.__init__(self, embedding, vocab_size, hparams)

        # Make RNN cells
        with tf.variable_scope(self.variable_scope):
            if cell_fw is not None:
                self._cell_fw = cell_fw
            else:
                self._cell_fw = layers.get_rnn_cell(self._hparams.rnn_cell_fw)

            if cell_bw is not None:
                self._cell_bw = cell_bw
            elif self.hparams.rnn_cell_share_config:
                self._cell_bw = layers.get_rnn_cell(self._hparams.rnn_cell_fw)
            else:
                self._cell_bw = layers.get_rnn_cell(self._hparams.rnn_cell_bw)

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        Returns:
            .. code-block:: python

                {
                    "rnn_cell_fw": default_rnn_cell_hparams(),
                    "rnn_cell_bw": default_rnn_cell_hparams(),
                    "rnn_cell_share_config": True
                    "use_embedding": True,
                    "embedding": default_embedding_hparams(),
                    "name": "unidirectional_rnn_encoder"
                }

            Here:

            "rnn_cell_fw" : dict
                A dictionary of hyperparameters of the forward RNN cell.
                Ignored if :attr:`cell_fw` is given when constructing
                the encoder.

                The default value is defined in
                :meth:`~txtgen.core.layers.default_rnn_cell_hparams`.

            "rnn_cell_bw" : dict
                A dictionary of hyperparameters of the backward RNN cell.
                Ignored if :attr:`cell_bw` is given when constructing
                the encoder, or if :attr:`"rnn_cell_share_config"` is `True`.

                The default value is defined in
                :meth:`~txtgen.core.layers.default_rnn_cell_hparams`.

            "rnn_cell_share_config" : bool
                Whether share hyperparameters of the backward cell with the
                forward cell.

                If `True` (default), :attr:`"rnn_cell_bw"` is ignored.

            "use_embedding" : bool
                Whether token embedding is used.

                If `True` (default), input to the encoder should contain integer
                indexes and will be used to look up the embedding vectors.
                If `False`, the input is directly fed into the RNN to encode.

            "embedding" : dict
                A dictionary of token embedding hyperparameters for
                embedding initialization.

                Ignored if :attr:`"use_embedding"` is `False`, or if
                :attr:`embedding` is given and is `Variable` when
                constructing the decoder.

                If :attr:`embedding` is given and is a Tensor or array, the
                "dim" and "initializer" specs in the embedding hyperparameterrs
                are ignored.

                The default value is defined in
                :meth:`~txtgen.core.layers.default_embedding_hparams`.

            "name" : str
                Name of the encoder
        """
        hparams = RNNEncoderBase.default_hparams()
        hparams["rnn_cell_fw"] = layers.default_rnn_cell_hparams()
        hparams["rnn_cell_share_config"] = True
        hparams["rnn_cell_bw"] = layers.default_rnn_cell_hparams()
        hparams["name"] = "bidirectional_rnn_encoder"
        return hparams

    def _build(self, inputs, sequence_length=None,
               initial_state_fw=None, initial_state_bw=None,
               **kwargs):
        """Encodes the inputs.

        Args:
            inputs: If embedding is enabled, this must be a 2D Tensor of shape
                `[batch_size, max_time]` containing input sequences of integer
                token indexes. Otherwise, this must be a 3D Tensor of shape
                `[batch_size, max_time, dim]`. The first two dimensions
                `batch_size` and `max_time` may be exchanged if
                `time_major=True` is specified.
            sequence_length (int list or 1D Tensor, optional): Sequence lengths
                of the batch inputs. Used to copy-through state and zero-out
                outputs when past a batch element's sequence length.
            initial_state (optional): Initial state of the RNN.
            **kwargs: Optional keyword arguments of
                :tf_main:`tf.nn.dynamic_rnn <nn/dynamic_rnn>`,
                such as `time_major`, `dtype`, etc.

        Returns:
            Outputs and final state of the encoder.
        """
        #TODO(zhiting): add docs of 'Returns'
        if self._embedding is not None:
            embedded_inputs = tf.nn.embedding_lookup(self._embedding, inputs)
        else:
            embedded_inputs = inputs

        no_initial_state = initial_state_fw is None and initial_state_bw is None
        if ('dtype' not in kwargs) and no_initial_state:
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=self._cell_fw,
                cell_bw=self._cell_bw,
                inputs=embedded_inputs,
                sequence_length=sequence_length,
                initial_state_fw=initial_state_fw,
                initial_state_bw=initial_state_bw,
                dtype=tf.float32,
                **kwargs)
        else:
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=self._cell_fw,
                cell_bw=self._cell_bw,
                inputs=embedded_inputs,
                sequence_length=sequence_length,
                initial_state_fw=initial_state_fw,
                initial_state_bw=initial_state_bw,
                **kwargs)

        if not self._built:
            self._add_internal_trainable_variables()
            # Add trainable variables of cells which may be constructed
            # externally.
            self._add_trainable_variable(
                layers.get_rnn_cell_trainable_variables(self._cell_fw))
            self._add_trainable_variable(
                layers.get_rnn_cell_trainable_variables(self._cell_bw))
            self._built = True

        return outputs, output_states

    @staticmethod
    def concat_outputs(outputs):
        """Concats the outputs of the bidirectional encoder into a single
        tensor.
        """
        return tf.concat(outputs, 2)

    @property
    def cell_fw(self):
        """The forward RNN cell.
        """
        return self._cell_fw

    @property
    def cell_bw(self):
        """The backward RNN cell.
        """
        return self._cell_bw

    @property
    def state_size_fw(self):
        """The state size of the forward encoder cell.

        Same as :attr:`encoder.cell_fw.state_size`.
        """
        return self.cell_fw.state_size

    @property
    def state_size_bw(self):
        """The state size of the backward encoder cell.

        Same as :attr:`encoder.cell_bw.state_size`.
        """
        return self.cell_bw.state_size
