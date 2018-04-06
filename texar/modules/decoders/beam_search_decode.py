#
"""
Beam search decoding for RNN decoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.seq2seq import dynamic_decode
from tensorflow.contrib.seq2seq import BeamSearchDecoder

from texar.modules.decoders.rnn_decoder_base import RNNDecoderBase

# pylint: disable=too-many-arguments, protected-access, too-many-locals

__all__ = [
    "beam_search_decode"
]

def beam_search_decode(decoder_or_cell,
                       embedding,
                       start_tokens,
                       end_token,
                       beam_width,
                       initial_state=None,
                       output_layer=None,
                       length_penalty_weight=0.0,
                       max_decoding_length=None,
                       output_time_major=False,
                       **kwargs):
    """Performs BeamSearch sampling decoding.

    Args:
        decoder_or_cell: An instance of
            :class:`~texar.modules.decoders.rnn_decoder_base.RNNDecoderBase`,
            or an instance of :tf_main:`RNNCell <contrib/rnn/RNNCell>`.
        embedding: A callable that takes a vector tensor of `ids` (argmax ids),
            or the :attr:`params` argument for
            :tf_main:`tf.nn.embedding_lookup <nn/embedding_lookup>`.
        start_tokens: `int32` vector shaped `[batch_size]`, the start tokens.
        end_token: `int32` scalar, the token that marks end of decoding.
        beam_width: Python integer, the number of beams.
        initial_state (optional): Initial state of decoding.
            If `None` (default), zero state is used.
        output_layer: (optional) An instance of `tf.layers.Layer` to apply
            to the RNN output prior to storing the result or sampling. If
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
            <contrib/seq2seq/dynamic_decode>`. Argument `maximum_iterations`
            is set to :attr:`max_decoding_length`.

    Returns:
        outputs: An instance of :tf_main:`FinalBeamSearchDecoderOutput
            <contrib/seq2seq/FinalBeamSearchDecoderOutput>`.
        final_state: An instance of :tf_main:`BeamSearchDecoderState
            <contrib/seq2seq/BeamSearchDecoderState>`.
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

    if initial_state is None:
        initial_state = cell.zero_state(batch_size * beam_width, tf.float32)
    if output_layer is None and isinstance(decoder_or_cell, RNNDecoderBase):
        output_layer = decoder_or_cell.output_layer

    beam_docoder = BeamSearchDecoder(
        cell=cell,
        embedding=embedding,
        start_tokens=start_tokens,
        end_token=end_token,
        initial_state=initial_state,
        beam_width=beam_width,
        output_layer=output_layer,
        length_penalty_weight=length_penalty_weight)

    if 'maximum_iterations' in kwargs:
        raise ValueError('Use `max_decoding_length` to set the maximum '
                         'allowed number of decoding steps.')
    outputs, final_state, _ = dynamic_decode(
        decoder=beam_docoder,
        output_time_major=output_time_major,
        maximum_iterations=max_decoding_length,
        **kwargs)

    return outputs, final_state
