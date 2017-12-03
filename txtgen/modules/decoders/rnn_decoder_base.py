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

from txtgen.modules.module_base import ModuleBase
from txtgen.modules.decoders import rnn_decoder_helpers
from txtgen.core import layers, utils
from txtgen import context

# pylint: disable=not-context-manager, too-many-arguments

__all__ = [
    "RNNDecoderBase"
]

class RNNDecoderBase(ModuleBase, TFDecoder):
    """Base class inherited by all RNN decoder classes.

    See :class:`~txtgen.modules.BasicRNNDecoder` for the argumenrts.
    """

    def __init__(self,
                 cell=None,
                 embedding=None,
                 vocab_size=None,
                 hparams=None,
                 output_layer=None):
        ModuleBase.__init__(self, hparams)

        self._helper = None
        self._initial_state = None
        self._output_layer = output_layer

        # Make rnn cell
        with tf.variable_scope(self.variable_scope):
            if cell is not None:
                self._cell = cell
            else:
                self._cell = layers.get_rnn_cell(self._hparams.rnn_cell)

        # Make embedding
        if vocab_size is None:
            if not self._hparams.use_embedding or embedding is None:
                raise ValueError(
                    "`vocab_size` is required if embedding is not enabled or "
                    "`embedding` is None.")

        self._embedding = None
        if self._hparams.use_embedding:
            if isinstance(embedding, tf.Variable):
                self._embedding = embedding
            else:
                self._embedding = layers.get_embedding(
                    self._hparams.embedding, embedding, vocab_size,
                    self.variable_scope)
            if self._hparams.embedding.trainable:
                self._add_trainable_variable(self._embedding)
            self._vocab_size = self._embedding.get_shape().as_list()[0]
        else:
            self._vocab_size = vocab_size

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        The hyperparameters have the same structure as in
        :meth:`~txtgen.modules.BasicRNNDecoder.default_hparams` of
        :class:`~txtgen.modules.BasicRNNDecoder`, except that the default
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


    @property
    def batch_size(self):
        return self._helper.batch_size

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
