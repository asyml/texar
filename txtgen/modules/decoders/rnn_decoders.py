"""
Various RNN decoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import tensorflow as tf
from tensorflow.contrib.seq2seq import BasicDecoderOutput, AttentionWrapperState

from tensorflow.contrib.seq2seq import AttentionWrapper
from tensorflow.python.framework import tensor_shape, dtypes # pylint: disable=E0611

from txtgen.modules.decoders.rnn_decoder_base import RNNDecoderBase
from txtgen.core.utils import get_instance

# pylint: disable=too-many-arguments
class AttentionDecoderOutput(collections.namedtuple("DecoderOutput", \
    ["logits", "predicted_ids", "cell_output", "attention_scores", "attention_context"])):
    """Augmented decoder output that also includes the attention scores.
    """
    pass

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
        #next_state should be cell_state directly,
        #according to function next_inouts
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
                 # mode, #mode, train/eval
                 vocab_size,
                 n_hidden, # num_units: The depth of the attention mechanism.
                 attention_keys, #encoder_output, [batch_size, max_length, ...]
                 attention_values, #encoder_output,
                 attention_values_length,
                 reverse_scores_lengths=None,
                 name='attention_rnn_decoder',
                 cell=None,
                 embedding=None,
                 embedding_trainable=True,
                 hparams=None):
        RNNDecoderBase.__init__(self, cell, embedding, vocab_size, hparams)
        att_params = hparams['attention']
        attention_class = hparams['attention']['class'] #LuongAttention
        attention_kwargs = hparams['attention']['params']
        attention_kwargs['num_units'] = n_hidden
        attention_kwargs['memory_sequence_length'] = attention_values_length
        attention_kwargs['memory'] = attention_keys
        attention_modules = ['txtgen.custom', 'tensorflow.contrib.seq2seq']
        attention_mechanism = get_instance(attention_class, attention_kwargs, attention_modules)

        wrapper_params = hparams['attention']['wrapper_params']
        attn_cell = AttentionWrapper(self._cell, attention_mechanism, **wrapper_params)
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
        hparams['attention'] = {
            'class':'LuongAttention',
            'alignment_history':False,
            'params':{
                'num_units':512,
                'scale':False,
                'probability_fn':None, #by default, it's softmax
                'score_mask_value':float('-inf'),
                'name':'LuongAttention'
            },
            'wrapper_params':{
                'attention_layer_size':None,
                'alignment_history':False,
                'cell_input_fn':None,
                'output_attention':True,
                'initial_cell_state':None
            }
        }
        return hparams
    def initialize(self, name=None):
        helper_init = self._helper.initialize()
        return [helper_init[0], helper_init[1], self._initial_state]

    def step(self, time, inputs, state, name=None):
        cell_outputs, cell_state = self._cell(inputs, state)
        wrapper_outputs, wrapper_state = self._cell(inputs, state)

        #cell_state is AttentionWrapperState
        cell_state = wrapper_state.cell_state
        attention_scores = wrapper_state.alignments
        attention_context = wrapper_state.attention

        logits = tf.contrib.layers.fully_connected(
            inputs=cell_outputs, num_outputs=self._vocab_size)
        sample_ids = self._helper.sample(
            time=time, outputs=logits, state=cell_state)
        (finished, next_inputs, next_state) = self._helper.next_inputs(
            time=time,
            outputs=logits,
            state=wrapper_state,
            sample_ids=sample_ids)
        # there should be some problem
        outputs = AttentionDecoderOutput(logits, sample_ids, \
                                         wrapper_outputs, attention_scores, attention_context)
        return (outputs, next_state, next_inputs, finished)

    def finalize(self, outputs, final_state, sequence_lengths):
        return outputs, final_state


    @property
    def output_size(self):
        statesize = self.cell.state_size
        return AttentionDecoderOutput(
            logits=self._vocab_size,
            predicted_ids=tensor_shape.TensorShape([]),
            cell_output=self.cell._cell.output_size,
            attention_scores=statesize.alignments,
            attention_context=statesize.attention
            )

    @property
    def output_dtype(self):
        return AttentionDecoderOutput(
            logits=dtypes.float32,
            predicted_ids=dtypes.int32,
            cell_output=dtypes.float32,
            attention_scores=dtypes.float32,
            attention_context=dtypes.float32,
            )
