#
"""
Base class for RNN decoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.seq2seq import Decoder as TFDecoder
from tensorflow.contrib.seq2seq import dynamic_decode
from tensorflow.python.framework import tensor_shape
from tensorflow.python.util import nest

from texar.modules.module_base import ModuleBase
from texar.modules.decoders import rnn_decoder_helpers
from texar.core import layers, utils
from texar import context

# pylint: disable=not-context-manager, too-many-arguments

__all__ = [
    "RNNDecoderBase"
]

class RNNDecoderBase(ModuleBase, TFDecoder):
    """Base class inherited by all RNN decoder classes.

    See :class:`~texar.modules.BasicRNNDecoder` for the argumenrts.
    """

    def __init__(self,
                 cell=None,
                 embedding=None,
                 vocab_size=None,
                 output_layer=None,
                 hparams=None):
        ModuleBase.__init__(self, hparams)

        self._helper = None
        self._initial_state = None

        # Make rnn cell
        with tf.variable_scope(self.variable_scope):
            if cell is not None:
                self._cell = cell
            else:
                self._cell = layers.get_rnn_cell(self._hparams.rnn_cell)

        # Make embedding
        if vocab_size is None:
            if self._hparams.use_embedding and embedding is None:
                raise ValueError(
                    "`vocab_size` is required if embedding is used and"
                    "`embedding` is None.")

        self._vocab_size = vocab_size
        self._embedding = None
        if self._hparams.use_embedding:
            if isinstance(embedding, tf.Variable):
                self._embedding = embedding
            else:
                self._embedding = layers.get_embedding(
                    self._hparams.embedding, embedding, self._vocab_size,
                    self.variable_scope)
            if self._hparams.embedding.trainable:
                self._add_trainable_variable(self._embedding)
            if self._vocab_size is None:
                self._vocab_size = self._embedding.get_shape().as_list()[0]

        # Make the output layer
        self._output_layer = output_layer
        if self._output_layer is None:
            if self._vocab_size is None:
                raise ValueError(
                    "Output layer size cannot be inferred automatically. "
                    "Must specify either `vocab_size` or "
                    "`embedding` (if embedding is used).")
            with tf.variable_scope(self.variable_scope):
                self._output_layer = tf.layers.Dense(units=self._vocab_size)
        elif self._output_layer is not tf.identity:
            if not isinstance(self._output_layer, tf.layers.Layer):
                raise ValueError(
                    "`output_layer` must be either `tf.identity` or ",
                    "instance of `tf.layers.Layer`.")
            self._add_trainable_variable(self._output_layer.trainable_variables)

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        The hyperparameters have the same structure as in
        :meth:`~texar.modules.BasicRNNDecoder.default_hparams` of
        :class:`~texar.modules.BasicRNNDecoder`, except that the default
        "name" here is "rnn_decoder".
        """
        return {
            "rnn_cell": layers.default_rnn_cell_hparams(),
            "use_embedding": True,
            "embedding": layers.default_embedding_hparams(),
            "helper_train": rnn_decoder_helpers.default_helper_train_hparams(),
            "helper_infer": rnn_decoder_helpers.default_helper_infer_hparams(),
            "max_decoding_length_train": None,
            "max_decoding_length_infer": None,
            "name": "rnn_decoder"
        }


    def _build(self, helper, initial_state):    # pylint: disable=W0221
        """Performs decoding.

        Args:
            helper: An instance of `tf.contrib.seq2seq.Helper` that helps with
                the decoding process. For example, use an instance of
                `TrainingHelper` in training phase.
            initial_state: Initial state of decoding.

        Returns:
            `(outputs, final_state, sequence_lengths)`: `outputs` is an object
            containing the decoder output on all time steps, `final_state` is
            the cell state of the final time step, `sequence_lengths` is a
            Tensor of shape `[batch_size]`.
        """
        self._helper = helper
        self._initial_state = initial_state

        max_decoding_length_train = self._hparams.max_decoding_length_train
        if max_decoding_length_train is None:
            max_decoding_length_train = utils.MAX_SEQ_LENGTH
        max_decoding_length_infer = self._hparams.max_decoding_length_infer
        if max_decoding_length_infer is None:
            max_decoding_length_infer = utils.MAX_SEQ_LENGTH
        max_decoding_length = tf.cond(
            context.is_train(),
            lambda: max_decoding_length_train,
            lambda: max_decoding_length_infer)
        outputs, final_state, sequence_lengths = dynamic_decode(
            decoder=self, maximum_iterations=max_decoding_length)

        self._add_internal_trainable_variables()
        # Add trainable variables of `self._cell` which may be constructed
        # externally.
        self._add_trainable_variable(
            layers.get_rnn_cell_trainable_variables(self._cell))
        self._built = True

        return outputs, final_state, sequence_lengths

    def _rnn_output_size(self):
        size = self._cell.output_size
        if self._output_layer is tf.identity:
            return size
        else:
            # To use layer's compute_output_shape, we need to convert the
            # RNNCell's output_size entries into shapes with an unknown
            # batch size.  We then pass this through the layer's
            # compute_output_shape and read off all but the first (batch)
            # dimensions to get the output size of the rnn with the layer
            # applied to the top.
            output_shape_with_unknown_batch = nest.map_structure(
                lambda s: tensor_shape.TensorShape([None]).concatenate(s),
                size)
            layer_output_shape = self._output_layer._compute_output_shape(
                output_shape_with_unknown_batch)
            return nest.map_structure(lambda s: s[1:], layer_output_shape)

    @property
    def batch_size(self):
        return self._helper.batch_size

    @property
    def output_size(self):
        """Output size of one step.
        """
        raise NotImplementedError

    @property
    def output_dtype(self):
        """Types of output of one step.
        """
        raise NotImplementedError

    def initialize(self, name=None):
        # Inherits from TFDecoder
        # All RNN decoder classes must implement this
        raise NotImplementedError

    def step(self, time, inputs, state, name=None):
        # Inherits from TFDecoder
        # All RNN decoder classes must implement this
        raise NotImplementedError

    def finalize(self, outputs, final_state, sequence_lengths):
        # Inherits from TFDecoder
        # All RNN decoder classes must implement this
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
        """The state size of decoder cell.

        Same as :attr:`decoder.cell.state_size`.
        """
        return self.cell.state_size

    @property
    def vocab_size(self):
        """The vocab size.
        """
        return self._vocab_size
