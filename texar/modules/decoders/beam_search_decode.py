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
Beam search decoding for RNN decoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.seq2seq import \
    dynamic_decode, AttentionWrapperState, AttentionWrapper, \
    BeamSearchDecoder, tile_batch

from texar.modules.decoders.rnn_decoder_base import RNNDecoderBase
# pylint: disable=too-many-arguments, protected-access, too-many-locals
# pylint: disable=invalid-name

__all__ = [
    "beam_search_decode"
]

def _get_initial_state(initial_state,
                       tiled_initial_state,
                       cell,
                       batch_size,
                       beam_width,
                       dtype):
    if tiled_initial_state is None:
        if isinstance(initial_state, AttentionWrapperState):
            raise ValueError(
                '`initial_state` must not be an AttentionWrapperState. Use '
                'a plain cell state instead, which will be wrapped into an '
                'AttentionWrapperState automatically.')
        if initial_state is None:
            tiled_initial_state = cell.zero_state(batch_size * beam_width,
                                                  dtype)
        else:
            tiled_initial_state = tile_batch(initial_state,
                                             multiplier=beam_width)

    if isinstance(cell, AttentionWrapper) and \
            not isinstance(tiled_initial_state, AttentionWrapperState):
        zero_state = cell.zero_state(batch_size * beam_width, dtype)
        tiled_initial_state = zero_state.clone(cell_state=tiled_initial_state)

    return tiled_initial_state


def beam_search_decode(decoder_or_cell,
                       embedding,
                       start_tokens,
                       end_token,
                       beam_width,
                       initial_state=None,
                       tiled_initial_state=None,
                       output_layer=None,
                       length_penalty_weight=0.0,
                       max_decoding_length=None,
                       output_time_major=False,
                       **kwargs):
    """Performs beam search sampling decoding.

    Args:
        decoder_or_cell: An instance of
            subclass of :class:`~texar.modules.RNNDecoderBase`,
            or an instance of :tf_main:`RNNCell <contrib/rnn/RNNCell>`. The
            decoder or RNN cell to perform decoding.
        embedding: A callable that takes a vector tensor of indexes (e.g.,
            an instance of subclass of :class:`~texar.modules.EmbedderBase`),
            or the :attr:`params` argument for
            :tf_main:`tf.nn.embedding_lookup <nn/embedding_lookup>`.
        start_tokens: `int32` vector shaped `[batch_size]`, the start tokens.
        end_token: `int32` scalar, the token that marks end of decoding.
        beam_width (int): Python integer, the number of beams.
        initial_state (optional): Initial state of decoding. If `None`
            (default), zero state is used.

            The state must **not** be tiled with
            :tf_main:`tile_batch <contrib/seq2seq/tile_batch>`.
            If you have an already-tiled initial state, use
            :attr:`tiled_initial_state` instead.

            In the case of attention RNN decoder, `initial_state` must
            **not** be an :tf_main:`AttentionWrapperState
            <contrib/seq2seq/AttentionWrapperState>`. Instead, it must be a
            state of the wrapped `RNNCell`, which state will be wrapped into
            `AttentionWrapperState` automatically.

            Ignored if :attr:`tiled_initial_state` is given.
        tiled_initial_state (optional): Initial state that has been tiled
            (typicaly with :tf_main:`tile_batch <contrib/seq2seq/tile_batch>`)
            so that the batch dimension has size `batch_size * beam_width`.

            In the case of attention RNN decoder, this can be either a state
            of the wrapped `RNNCell`, or an `AttentionWrapperState`.

            If not given, :attr:`initial_state` is used.
        output_layer (optional): A :tf_main:`Layer <layers/Layer>` instance to
            apply to the RNN output prior to storing the result or sampling. If
            `None` and :attr:`decoder_or_cell` is a decoder, the decoder's
            output layer will be used.
        length_penalty_weight: Float weight to penalize length.
            Disabled with `0.0` (default).
        max_decoding_length (optional): A int scalar Tensor indicating the
            maximum allowed number of decoding steps. If `None` (default),
            decoding will continue until the end token is encountered.
        output_time_major (bool): If `True`, outputs are returned as
            time major tensors. If `False` (default), outputs are returned
            as batch major tensors.
        **kwargs: Other keyword arguments for :tf_main:`dynamic_decode
            <contrib/seq2seq/dynamic_decode>` except argument
            `maximum_iterations` which is set to :attr:`max_decoding_length`.

    Returns:
        A tuple `(outputs, final_state, sequence_length)`, where

        - outputs: An instance of :tf_main:`FinalBeamSearchDecoderOutput \
        <contrib/seq2seq/FinalBeamSearchDecoderOutput>`.
        - final_state: An instance of :tf_main:`BeamSearchDecoderState \
        <contrib/seq2seq/BeamSearchDecoderState>`.
        - sequence_length: A Tensor of shape `[batch_size]` containing \
        the lengths of samples.

    Example:

        .. code-block:: python

            ## Beam search with basic RNN decoder

            embedder = WordEmbedder(vocab_size=data.vocab.size)
            decoder = BasicRNNDecoder(vocab_size=data.vocab.size)

            outputs, _, _, = beam_search_decode(
                decoder_or_cell=decoder,
                embedding=embedder,
                start_tokens=[data.vocab.bos_token_id] * 100,
                end_token=data.vocab.eos_token_id,
                beam_width=5,
                max_decoding_length=60)

            sample_ids = sess.run(outputs.predicted_ids)
            sample_text = tx.utils.map_ids_to_strs(sample_id[:,:,0], data.vocab)
            print(sample_text)
            # [
            #   the first sequence sample .
            #   the second sequence sample .
            #   ...
            # ]

        .. code-block:: python

            ## Beam search with attention RNN decoder

            # Encodes the source
            enc_embedder = WordEmbedder(data.source_vocab.size, ...)
            encoder = UnidirectionalRNNEncoder(...)

            enc_outputs, enc_state = encoder(
                inputs=enc_embedder(data_batch['source_text_ids']),
                sequence_length=data_batch['source_length'])

            # Decodes while attending to the source
            dec_embedder = WordEmbedder(vocab_size=data.target_vocab.size, ...)
            decoder = AttentionRNNDecoder(
                memory=enc_outputs,
                memory_sequence_length=data_batch['source_length'],
                vocab_size=data.target_vocab.size)

            # Beam search
            outputs, _, _, = beam_search_decode(
                decoder_or_cell=decoder,
                embedding=dec_embedder,
                start_tokens=[data.vocab.bos_token_id] * 100,
                end_token=data.vocab.eos_token_id,
                beam_width=5,
                initial_state=enc_state,
                max_decoding_length=60)
    """
    if isinstance(decoder_or_cell, RNNDecoderBase):
        cell = decoder_or_cell._get_beam_search_cell(beam_width=beam_width)
    elif isinstance(decoder_or_cell, tf.contrib.rnn.RNNCell):
        cell = decoder_or_cell
    else:
        raise ValueError("`decoder` must be an instance of a subclass of "
                         "either `RNNDecoderBase` or `RNNCell`.")

    start_tokens = tf.convert_to_tensor(
        start_tokens, dtype=tf.int32, name="start_tokens")
    if start_tokens.get_shape().ndims != 1:
        raise ValueError("`start_tokens` must be a vector")
    batch_size = tf.size(start_tokens)

    initial_state = _get_initial_state(
        initial_state, tiled_initial_state, cell,
        batch_size, beam_width, tf.float32)

    if output_layer is None and isinstance(decoder_or_cell, RNNDecoderBase):
        output_layer = decoder_or_cell.output_layer

    def _decode():
        beam_docoder = BeamSearchDecoder(
            cell=cell,
            embedding=embedding,
            start_tokens=start_tokens,
            end_token=end_token,
            initial_state=initial_state,
            beam_width=beam_width,
            output_layer=None if output_layer is tf.identity else output_layer,
            length_penalty_weight=length_penalty_weight)

        if 'maximum_iterations' in kwargs:
            raise ValueError('Use `max_decoding_length` to set the maximum '
                             'allowed number of decoding steps.')
        outputs, final_state, _ = dynamic_decode(
            decoder=beam_docoder,
            output_time_major=output_time_major,
            maximum_iterations=max_decoding_length,
            **kwargs)

        return outputs, final_state, final_state.lengths

    if isinstance(decoder_or_cell, RNNDecoderBase):
        vs = decoder_or_cell.variable_scope
        with tf.variable_scope(vs, reuse=tf.AUTO_REUSE):
            return _decode()
    else:
        return _decode()
