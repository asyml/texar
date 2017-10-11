#
"""
Various RNN decoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.seq2seq import BasicDecoderOutput
from tensorflow.python.framework import tensor_shape, dtypes # pylint: disable=E0611

from txtgen.modules.decoders.rnn_decoder_base import RNNDecoderBase

# pylint: disable=too-many-arguments

class BasicRNNDecoder(RNNDecoderBase):
    """Basic RNN decoder that performs sampling at each step.

    See :class:`~txtgen.modules.decoders.RNNDecoderBase` for the arguments.
    """

    def __init__(self,
                 cell=None,
                 embedding=None,
                 vocab_size=None,
                 hparams=None):
        RNNDecoderBase.__init__(self, cell, embedding, vocab_size, hparams)

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        The hyperparameters have the same structure as in
        :meth:`~txtgen.modules.RNNDecoderBase.default_hparams`, except that
        the default "name" is "basic_rnn_decoder".
        """
        hparams = RNNDecoderBase.default_hparams()
        hparams["name"] = "basic_rnn_decoder"
        return hparams

    def initialize(self, name=None):
        return self._helper.initialize() + (self._initial_state,)

    def step(self, time, inputs, state, name=None):
        cell_outputs, cell_state = self._cell(inputs, state)
        logits = tf.contrib.layers.fully_connected(
            inputs=cell_outputs, num_outputs=self._vocab_size)
        sample_ids = self._helper.sample(
            time=time, outputs=logits, state=cell_state)
        (finished, next_inputs, next_state) = self._helper.next_inputs(
            time=time,
            outputs=logits,
            state=cell_state,
            sample_ids=sample_ids)
        outputs = BasicDecoderOutput(logits, sample_ids)
        return (outputs, next_state, next_inputs, finished)

    def finalize(self, outputs, final_state, sequence_lengths):
        return outputs, final_state

    @property
    def output_size(self):
        return BasicDecoderOutput(
            rnn_output=self._vocab_size,
            sample_id=tensor_shape.TensorShape([]))

    @property
    def output_dtype(self):
        return BasicDecoderOutput(
            rnn_output=dtypes.float32, sample_id=dtypes.int32)

class AttentionRNNDecoder(RNNDecoderBase):
    """Basic RNN decoder that performs sampling at each step.
    See :class:`~txtgen.modules.decoders.RNNDecoderBase` for the arguments.
    """

    def __init__(self,  # pylint: disable=too-many-arguments
                 mode,
                 vocab_size,
                 n_hidden, # num_units: The depth of the attention mechanism.
                 attention_keys, #encoder_output, [batch_size, max_length, ...]
                 attention_values, #encoder_output,
                 attention_values_length,
                 attention_fn,
                 reverse_scores_lengths=None,
                 name = 'attention_rnn_decoder'
                 cell=None,
                 embedding=None,
                 embedding_trainable=True,
                 hparams=None):
        AttentionRNNDecoder.__init__(self, cell, embedding, vocab_size, hparams)
        att_params = hparams['attention']
        attention_class = get_class(hparams['attention']['class']) #LuongAttention
        attention_mechanism = attention_class(n_hidden, attention_keys, attention_values_length)
        attn_cell = get_class('AttentionWrapper')(self._cell, attention_mechanism) # more parameters can be added here
        self._cell = attn_cell

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.
        The hyperparameters have the same structure as in
        :meth:`txtgen.modules.RNNDecoderBase.default_hparams`, except that
        the default "name" is "basic_rnn_decoder".
        """
        hparams = RNNDecoderBase.default_hparams()
        hparams["name"] = "attention_rnn_decoder"
        return hparams

    def initialize(self, name=None):
        return self._helper.initialize() + (self._initial_state,)

    def step(self, time, inputs, state, name=None):
        cell_outputs, cell_state = self._cell(inputs, state)
        logits = tf.contrib.layers.fully_connected(
            inputs=cell_outputs, num_outputs=self._vocab_size)
        sample_ids = self._helper.sample(
            time=time, outputs=logits, state=cell_state)
        (finished, next_inputs, next_state) = self._helper.next_inputs(
            time=time,
            outputs=logits,
            state=cell_state,
            sample_ids=sample_ids)
        outputs = BasicDecoderOutput(logits, sample_ids)
        return (outputs, next_state, next_inputs, finished)

    def finalize(self, outputs, final_state, sequence_lengths):
        return outputs, final_state

    @property
    def output_size(self):
        return BasicDecoderOutput(
            rnn_output=self._vocab_size,
            sample_id=tensor_shape.TensorShape([]))

    @property
    def output_dtype(self):
        return BasicDecoderOutput(
            rnn_output=dtypes.float32, sample_id=dtypes.int32)
