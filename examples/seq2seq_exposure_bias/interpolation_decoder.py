# Copyright 2018 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Interpolation Decoder is used for interpolation algorithm
which stores one more variable in 'state' recording the
decoded ids(state: [decoded_ids, rnn_state]).
"""

# pylint: disable=no-name-in-module, too-many-arguments, too-many-locals
# pylint: disable=not-context-manager, protected-access, invalid-name

import tensorflow as tf

from texar.tf.modules.decoders.rnn_decoders import \
    AttentionRNNDecoder, AttentionRNNDecoderOutput


class InterpolationDecoder(AttentionRNNDecoder):
    """
    Basicly the same as AttentionRNNDecoder except one
    more variable except rnn_state in 'state' recording the
    decoded ids(state: [decoded_ids, rnn_state])

    Args:
        memory: The memory to query, e.g., the output of an RNN encoder. This
            tensor should be shaped `[batch_size, max_time, dim]`.
        memory_sequence_length (optional): A tensor of shape `[batch_size]`
            containing the sequence lengths for the batch
            entries in memory. If provided, the memory tensor rows are masked
            with zeros for values past the respective sequence lengths.
        cell (RNNCell, optional): An instance of `RNNCell`. If `None`, a cell
            is created as specified in :attr:`hparams`.
        cell_dropout_mode (optional): A Tensor taking value of
            :tf_main:`tf.estimator.ModeKeys <estimator/ModeKeys>`, which
            toggles dropout in the RNN cell (e.g., activates dropout in
            TRAIN mode). If `None`, :func:`~texar.tf.global_mode` is used.
            Ignored if :attr:`cell` is given.
        vocab_size (int, optional): Vocabulary size. Required if
            :attr:`output_layer` is `None`.
        output_layer (optional): An instance of
            :tf_main:`tf.layers.Layer <layers/Layer>`, or
            :tf_main:`tf.identity <identity>`. Apply to the RNN cell
            output to get logits. If `None`, a dense layer
            is used with output dimension set to :attr:`vocab_size`.
            Set `output_layer=tf.identity` if you do not want to have an
            output layer after the RNN cell outputs.
        cell_input_fn (callable, optional): A callable that produces RNN cell
            inputs. If `None` (default), the default is used:
            `lambda inputs, attention: tf.concat([inputs, attention], -1)`,
            which cancats regular RNN cell inputs with attentions.
        hparams (dict, optional): Hyperparameters. Missing
            hyperparamerter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter sturcture and
            default values.
    """
    def __init__(self,
                 memory,
                 memory_sequence_length=None,
                 cell=None,
                 cell_dropout_mode=None,
                 vocab_size=None,
                 output_layer=None,
                 cell_input_fn=None,
                 hparams=None):
        AttentionRNNDecoder.__init__(
            self, memory, memory_sequence_length, cell, cell_dropout_mode,
            vocab_size, output_layer, cell_input_fn, hparams)

    def initialize(self, name=None):
        init = AttentionRNNDecoder.initialize(self, name)

        batch_size = tf.shape(init[0])[0]

        # decoded_ids can be initialized as any arbitrary value
        # because it will be assigned later in decoding
        initial_decoded_ids = tf.ones((batch_size, 60), dtype=tf.int32)

        initial_rnn_state = init[2]
        initial_state = [initial_decoded_ids, initial_rnn_state]
        init[2] = initial_state

        return init

    def step(self, time, inputs, state, name=None):
        # Basicly the same as in AttentionRNNDecoder except considering
        # about the different form of 'state'(decoded_ids, rnn_state)

        wrapper_outputs, wrapper_state = self._cell(inputs, state[1])
        decoded_ids = state[0]

        logits = self._output_layer(wrapper_outputs)

        sample_ids = self._helper.sample(
            time=time, outputs=logits, state=[decoded_ids, wrapper_state])

        attention_scores = wrapper_state.alignments
        attention_context = wrapper_state.attention
        outputs = AttentionRNNDecoderOutput(
            logits, sample_ids, wrapper_outputs,
            attention_scores, attention_context)

        return (outputs, wrapper_state)

    def next_inputs(self, time, outputs, state):
        (finished, next_inputs, next_state) = self._helper.next_inputs(
            time=time,
            outputs=outputs.logits,
            state=[state[0], state],
            sample_ids=outputs.sample_id)
        return (finished, next_inputs, next_state)
