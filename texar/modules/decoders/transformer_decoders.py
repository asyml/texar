# Copyright 2019 The Texar Authors. All Rights Reserved.
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
Transformer decoder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=no-name-in-module, too-many-arguments, too-many-locals
# pylint: disable=invalid-name, too-many-instance-attributes,
# pylint: disable=too-many-branches, redefined-variable-type

import collections

import tensorflow as tf
from tensorflow.contrib.seq2seq import Decoder as TFDecoder
from tensorflow.contrib.seq2seq import dynamic_decode

from texar.core import layers
from texar.module_base import ModuleBase
from texar.modules.networks.networks import FeedForwardNetwork
from texar.modules.encoders.transformer_encoders import \
    default_transformer_poswise_net_hparams
from texar.modules.encoders.multihead_attention import \
    MultiheadAttentionEncoder
from texar.modules.decoders.rnn_decoder_base import _make_output_layer
from texar.modules.decoders import tf_helpers as tx_helper
from texar.utils import beam_search, transformer_attentions as attn
from texar.utils.shapes import shape_list
from texar.utils.mode import is_train_mode


__all__ = [
    "TransformerDecoderOutput",
    "TransformerDecoder"
]


class TransformerDecoderOutput(collections.namedtuple(
        "TransformerDecoderOutput",
        ("logits", "sample_id"))):
    """The output of :class:`TransformerDecoder`.

    Attributes:
        logits: A float Tensor of shape
            `[batch_size, max_time, vocab_size]` containing the logits.
        sample_id: An int Tensor of shape `[batch_size, max_time]`
            containing the sampled token indexes.
    """


class TransformerDecoder(ModuleBase, TFDecoder):
    """Transformer decoder that applies multi-head self-attention for
    sequence decoding.

    It is a stack of :class:`~texar.modules.encoders.MultiheadAttentionEncoder`,
    :class:`~texar.modules.FeedForwardNetwork` and residual connections.

    Args:
        vocab_size (int, optional): Vocabulary size. Required if
            :attr:`output_layer` is `None`.
        output_layer (optional): An output layer that transforms cell output
            to logits. This can be:

            - A callable layer, e.g., an instance \
            of :tf_main:`tf.layers.Layer <layers/Layer>`.
            - A tensor. A dense layer will be created using the tensor \
            as the kernel weights. The bias of the dense layer is determined by\
            `hparams.output_layer_bias`. This can be used to tie the output \
            layer with the input embedding matrix, as proposed in \
            https://arxiv.org/pdf/1608.05859.pdf
            - `None`. A dense layer will be created based on attr:`vocab_size`\
            and `hparams.output_layer_bias`.
            - If no output layer in the end is needed, set \
            `(vocab_size=None, output_layer=tf.identity)`.

    .. document private functions
    .. automethod:: _build
    """

    def __init__(self,
                 vocab_size=None,
                 output_layer=None,
                 hparams=None):
        ModuleBase.__init__(self, hparams)


        with tf.variable_scope(self.variable_scope):
            if self._hparams.initializer:
                tf.get_variable_scope().set_initializer(
                    layers.get_initializer(self._hparams.initializer))

            # Make the output layer
            self._output_layer, self._vocab_size = _make_output_layer(
                output_layer, vocab_size, self._hparams.output_layer_bias,
                self.variable_scope)

            # Make attention and poswise networks
            self.multihead_attentions = {
                'self_att': [],
                'encdec_att': []
            }
            self.poswise_networks = []
            for i in range(self._hparams.num_blocks):
                layer_name = 'layer_{}'.format(i)
                with tf.variable_scope(layer_name):
                    with tf.variable_scope("self_attention"):
                        multihead_attention = MultiheadAttentionEncoder(
                            self._hparams.multihead_attention)
                        self.multihead_attentions['self_att'].append(
                            multihead_attention)

                    if self._hparams.dim != \
                            multihead_attention.hparams.output_dim:
                        raise ValueError('The output dimenstion of '
                                         'MultiheadEncoder should be equal '
                                         'to the dim of TransformerDecoder')

                    with tf.variable_scope('encdec_attention'):
                        multihead_attention = MultiheadAttentionEncoder(
                            self._hparams.multihead_attention)
                        self.multihead_attentions['encdec_att'].append(
                            multihead_attention)

                    if self._hparams.dim != \
                            multihead_attention.hparams.output_dim:
                        raise ValueError('The output dimenstion of '
                                         'MultiheadEncoder should be equal '
                                         'to the dim of TransformerDecoder')

                    pw_net = FeedForwardNetwork(
                        hparams=self._hparams['poswise_feedforward'])
                    final_dim = pw_net.hparams.layers[-1]['kwargs']['units']
                    if self._hparams.dim != final_dim:
                        raise ValueError(
                            'The output dimenstion of '
                            '"poswise_feedforward" should be equal '
                            'to the "dim" of TransformerDecoder.')
                    self.poswise_networks.append(pw_net)

            # Built in _build()
            self.context = None
            self.context_sequence_length = None
            self.embedding = None
            self._helper = None
            self._cache = None
            self.max_decoding_length = None

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                # Same as in TransformerEncoder
                "num_blocks": 6,
                "dim": 512,
                "embedding_dropout": 0.1,
                "residual_dropout": 0.1,
                "poswise_feedforward": default_transformer_poswise_net_hparams,
                "multihead_attention": {
                    'name': 'multihead_attention',
                    'num_units': 512,
                    'output_dim': 512,
                    'num_heads': 8,
                    'dropout_rate': 0.1,
                    'output_dim': 512,
                    'use_bias': False,
                },
                "initializer": None,
                "name": "transformer_decoder"
                # Additional for TransformerDecoder
                "embedding_tie": True,
                "output_layer_bias": False,
                "max_decoding_length": int(1e10),
            }

        Here:

        "num_blocks" : int
            Number of stacked blocks.

        "dim" : int
            Hidden dimension of the encoder.

        "embedding_dropout" : float
            Dropout rate of the input word and position embeddings.

        "residual_dropout" :  float
            Dropout rate of the residual connections.

        "poswise_feedforward" : dict
            Hyperparameters for a feed-forward network used in residual
            connections.
            Make sure the dimension of the output tensor is equal to `dim`.

            See :func:`~texar.modules.default_transformer_poswise_net_hparams`
            for details.

        "multihead_attention" : dict
            Hyperparameters for the multihead attention strategy.
            Make sure the `output_dim` in this module is equal to `dim`.

            See :func:`~texar.modules.MultiheadAttentionEncoder.default_hparams`
            for details.

        "initializer" : dict, optional
            Hyperparameters of the default initializer that initializes
            variables created in this module.
            See :func:`~texar.core.get_initializer` for details.

        "output_layer_bias" : bool
            Whether to use bias to the output layer.
            Used only if :attr:`output_layer` is `None` when constructing
            the class instance.

        "max_decoding_length" : int
            The maximum allowed number of decoding steps.
            Set to a very large number of avoid the length constraint.
            Ignored if provided in :meth:`_build` or
            "train_greedy" decoding is used.

            Length penalty coefficient. Refer to
            https://arxiv.org/abs/1609.08144 for more details.

        "name" : str
            Name of the module.
        """
        return {
            "num_blocks": 6,
            "dim": 512,
            "embedding_tie": True,
            "output_layer_bias": False,
            "max_decoding_length": int(1e10),
            "embedding_dropout": 0.1,
            "residual_dropout": 0.1,
            "poswise_feedforward": default_transformer_poswise_net_hparams(),
            'multihead_attention': {
                'name': 'multihead_attention',
                'num_units': 512,
                'num_heads': 8,
                'dropout_rate': 0.1,
                'output_dim': 512,
                'use_bias': False,
            },
            "initializer": None,
            "name": "transformer_decoder",
        }

    def _inputs_to_outputs(self, inputs, cache):
        """The function is called in dynamic decoding.

        `inputs` should be of shape `[batch_size, dim]`.

        Returns outputs (i.e. logits) of shape `[batch_size, vocab_size]`
        and updated cache.
        """
        outputs = self._self_attention_stack(
            tf.expand_dims(inputs, axis=1),
            memory=cache.get('memory'),
            cache=cache,
        )
        outputs = self._output_layer(outputs)
        outputs = tf.squeeze(outputs, axis=[1])
        return outputs, cache

    def _input_ids_to_outputs(self, input_ids, step, cache):
        """The function is called in beam-search decoding.

        `inputs` should be of shape `[batch_size]`.

        Returns outputs (i.e. logits) of shape `[batch_size, vocab_size]`
        and updated cache.
        """
        _batch_size = shape_list(input_ids)[0]
        times = tf.ones([_batch_size], dtype=tf.int32) * step
        inputs = self.embedding(input_ids, times)

        outputs = self._self_attention_stack(
            tf.expand_dims(inputs, axis=1),
            memory=cache.get('memory'),
            cache=cache,
        )
        outputs = self._output_layer(outputs)
        outputs = tf.squeeze(outputs, axis=[1])
        return outputs, cache

    def _build(self,  # pylint: disable=arguments-differ, too-many-statements
               decoding_strategy='train_greedy',
               inputs=None,
               memory=None,
               memory_sequence_length=None,
               memory_attention_bias=None,
               beam_width=None,
               length_penalty=0.,
               start_tokens=None,
               end_token=None,
               context=None,
               context_sequence_length=None,
               softmax_temperature=None,
               max_decoding_length=None,
               impute_finished=False,
               embedding=None,
               helper=None,
               mode=None):
        """Performs decoding.

        The interface is mostly the same with that of RNN decoders
        (see :meth:`~texar.modules.RNNDecoderBase._build`). The main difference
        is that, here, `sequence_length` is not needed, and continuation
        generation is additionally supported.

        The function provides **3 ways** to specify the decoding method, with
        varying flexibility:

        1. The :attr:`decoding_strategy` argument.

            - **"train_greedy"**: decoding in teacher-forcing fashion (i.e.,
              feeding ground truth to decode the next step), and for each step
              sample is obtained by taking the `argmax` of logits.
              Argument :attr:`inputs` is required for this strategy.
            - **"infer_greedy"**: decoding in inference fashion (i.e., feeding
              `generated` sample to decode the next step), and for each step
              sample is obtained by taking the `argmax` of logits.
              Arguments :attr:`(start_tokens, end_token)` are
              required for this strategy, and argument
              :attr:`max_decoding_length` is optional.
            - **"infer_sample"**: decoding in inference fashion, and for each
              step sample is obtained by `random sampling` from the logits.
              Arguments :attr:`(start_tokens, end_token)` are required for this
              strategy, and argument :attr:`max_decoding_length` is optional.

          This argument is used only when arguments :attr:`helper` and
          :attr:`beam_width` are both `None`.

        2. The :attr:`helper` argument: An instance of subclass of
           :class:`texar.modules.Helper`.
           This provides a superset of decoding strategies than above.
           The interface is the same as in RNN decoders.
           Please refer to :meth:`texar.modules.RNNDecoderBase._build` for
           detailed usage and examples.

           Note that, here, though using a
           :class:`~texar.modules.TrainingHelper` corresponds to the
           "train_greedy" strategy above and will get the same output results,
           the implementation is *slower* than
           directly setting `decoding_strategy="train_greedy"`.

           Argument :attr:`max_decoding_length` is optional.

        3. **Beam search**: set :attr:`beam_width` to use beam search decoding.
           Arguments :attr:`(start_tokens, end_token)` are required,
           and argument :attr:`max_decoding_length` is optional.

        Args:
            memory (optional): The memory to attend, e.g., the output of an RNN
                encoder. A Tensor of shape `[batch_size, memory_max_time, dim]`.
            memory_sequence_length (optional): A Tensor of shape `[batch_size]`
                containing the sequence lengths for the batch entries in
                memory. Used to create attention bias of
                :attr:`memory_attention_bias` is not given. Ignored if
                `memory_attention_bias` is provided.
            memory_attention_bias (optional): A Tensor of shape
                `[batch_size, num_heads, memory_max_time, dim]`.
                An attention bias typically sets the value of a padding
                position to a large negative value for masking. If not given,
                :attr:`memory_sequence_length` is used to automatically
                create an attention bias.
            inputs (optional): Input tensor for teacher forcing decoding, of
                shape `[batch_size, target_max_time, emb_dim]` containing the
                target sequence word embeddings.
                Used when :attr:`decoding_strategy` is set to "train_greedy".
            decoding_strategy (str): A string specifying the decoding
                strategy, including "train_greedy", "infer_greedy",
                "infer_sample".
                Different arguments are required based on the
                strategy. See above for details. Ignored if
                :attr:`beam_width` or :attr:`helper` is set.
            beam_width (int): Set to use beam search. If given,
                :attr:`decoding_strategy` is ignored.
            length_penalty (float): Length penalty coefficient used in beam
                search decoding. Refer to https://arxiv.org/abs/1609.08144
                for more details.
                It Should be larger if longer sentences are wanted.
            start_tokens (optional): An int Tensor of shape `[batch_size]`,
                containing the start tokens.
                Used when :attr:`decoding_strategy` = "infer_greedy" or
                "infer_sample", or :attr:`beam_width` is set.
                Ignored when context is set.
            end_token (optional): An int 0D Tensor, the token that marks end
                of decoding.
                Used when :attr:`decoding_strategy` = "infer_greedy" or
                "infer_sample", or :attr:`beam_width` is set.
            context (optional): An int Tensor of shape `[batch_size, length]`,
                containing the starting tokens for decoding.
                If context is set, the start_tokens will be ignored.
            context_sequence_length (optional): specify the length of context.
            softmax_temperature (optional): A float 0D Tensor, value to divide
                the logits by before computing the softmax. Larger values
                (above 1.0) result in more random samples. Must > 0. If `None`,
                1.0 is used.
                Used when :attr:`decoding_strategy` = "infer_sample"`.
            max_decoding_length (optional): An int scalar Tensor indicating
                the maximum allowed number of decoding steps.
                If `None` (default), use "max_decoding_length" defined in
                :attr:`hparams`. Ignored in "train_greedy" decoding.
            impute_finished (bool): If `True`, then states for batch
                entries which are marked as finished get copied through and
                the corresponding outputs get zeroed out.  This causes some
                slowdown at each time step, but ensures that the final state
                and outputs have the correct values and that backprop ignores
                time steps that were marked as finished. Ignored in
                "train_greedy" decoding.
            embedding (optional): Embedding used when
                "infer_greedy" or "infer_sample" `decoding_strategy`, or
                beam search, is used. This can be
                a callable or the `params` argument for
                :tf_main:`embedding_lookup <nn/embedding_lookup>`.
                If a callable, it can take a vector tensor of token `ids`,
                or take two arguments (`ids`, `times`), where `ids`
                is a vector tensor of token ids, and `times` is a vector tensor
                of time steps (i.e., position ids). The latter case can be used
                when attr:`embedding` is a combination of word embedding and
                position embedding.
            helper (optional): An instance of
                :tf_main:`Helper <contrib/seq2seq/Helper>` that defines the
                decoding strategy. If given, :attr:`decoding_strategy` is
                ignored.
            mode (optional): A tensor taking value in
                :tf_main:`tf.estimator.ModeKeys <estimator/ModeKeys>`, including
                `TRAIN`, `EVAL`, and `PREDICT`. Controls dropout mode.
                If `None` (default), :func:`texar.global_mode`
                is used.

        Returns:

            - For **"train_greedy"** decoding, returns an instance of \
            :class:`~texar.modules.TransformerDecoderOutput` which contains\
            `sample_id` and `logits`.

            - For **"infer_greedy"** and **"infer_sample"** decoding or\
            decoding with :attr:`helper`, returns\
            a tuple `(outputs, sequence_lengths)`, where `outputs` is an \
            instance of :class:`~texar.modules.TransformerDecoderOutput` as\
            in "train_greedy", and `sequence_lengths` is a Tensor of shape\
            `[batch_size]` containing the length of each sample.

            - For **beam search** decoding, returns a `dict` containing keys\
            "sample_id" and "log_prob".

                - **"sample_id"** is an int Tensor of shape \
                `[batch_size, max_time, beam_width]` containing generated\
                token indexes. `sample_id[:,:,0]` is the highest-probable \
                sample.
                - **"log_prob"** is a float Tensor of shape \
                `[batch_size, beam_width]` containing the log probability \
                of each sequence sample.
        """

        if memory is not None:
            if memory_attention_bias is None:
                if memory_sequence_length is None:
                    raise ValueError(
                        "`memory_sequence_length` is required if "
                        "`memory_attention_bias` is not given.")

                enc_padding = 1 - tf.sequence_mask(
                    memory_sequence_length, shape_list(memory)[1],
                    dtype=tf.float32)
                memory_attention_bias = attn.attention_bias_ignore_padding(
                    enc_padding)

        # record the context, which will be used in step function
        # for dynamic_decode
        if context is not None:
            start_tokens = context[:, 0]
            self.context = context[:, 1:]
            self.context_sequence_length = context_sequence_length - 1
        else:
            self.context = None

        self.embedding = embedding

        if helper is None and beam_width is None and \
                decoding_strategy == 'train_greedy':  # Teacher-forcing

            decoder_self_attention_bias = (
                attn.attention_bias_lower_triangle(
                    shape_list(inputs)[1]))

            decoder_output = self._self_attention_stack(
                inputs,
                memory,
                decoder_self_attention_bias=decoder_self_attention_bias,
                memory_attention_bias=memory_attention_bias,
                cache=None,
                mode=mode)
            logits = self._output_layer(decoder_output)
            preds = tf.to_int32(tf.argmax(logits, axis=-1))
            rets = TransformerDecoderOutput(
                logits=logits,
                sample_id=preds
            )

        else:
            if max_decoding_length is None:
                max_decoding_length = self._hparams.max_decoding_length
            self.max_decoding_length = max_decoding_length
            if beam_width is None:  # Inference-like decoding
                # Prepare helper
                if helper is None:
                    if decoding_strategy == "infer_greedy":
                        helper = tx_helper.GreedyEmbeddingHelper(
                            embedding, start_tokens, end_token)
                    elif decoding_strategy == "infer_sample":
                        helper = tx_helper.SampleEmbeddingHelper(
                            embedding, start_tokens, end_token,
                            softmax_temperature)
                    else:
                        raise ValueError(
                            "Unknown decoding strategy: {}".format(
                                decoding_strategy))
                self._helper = helper

                self._cache = self._init_cache(memory, memory_attention_bias,
                                               beam_search_decoding=False)
                if context is not None:
                    self.context = tf.pad(
                        self.context,
                        [[0, 0],
                         [0, max_decoding_length - shape_list(self.context)[1]]]
                    )

                outputs, _, sequence_lengths = dynamic_decode(
                    decoder=self,
                    impute_finished=impute_finished,
                    maximum_iterations=max_decoding_length,
                    output_time_major=False,
                    scope=self.variable_scope)

                if context is not None:
                    # Here the length of sample_id will be larger than that
                    # of logit by 1, because there will be a additional
                    # start_token in the returned sample_id.
                    # the start_id should be the first token of the
                    # given context
                    outputs = TransformerDecoderOutput(
                        logits=outputs.logits,
                        sample_id=tf.concat(
                            [tf.expand_dims(start_tokens, 1),
                             outputs.sample_id],
                            axis=1
                        )
                    )
                    sequence_lengths = sequence_lengths + 1
                rets = outputs, sequence_lengths

            else:  # Beam-search decoding
                # Ignore `decoding_strategy`; Assume `helper` is not set
                if helper is not None:
                    raise ValueError("Must not set 'beam_width' and 'helper' "
                                     "simultaneously.")
                _batch_size = shape_list(start_tokens)[0]
                self._cache = self._init_cache(memory, memory_attention_bias,
                                               beam_search_decoding=True,
                                               batch_size=_batch_size)

                # The output format is different when running beam search
                sample_id, log_prob = self._beam_decode(
                    start_tokens,
                    end_token,
                    beam_width=beam_width,
                    length_penalty=length_penalty,
                    decode_length=max_decoding_length,
                )
                rets = {
                    'sample_id': sample_id,
                    'log_prob': log_prob
                }

        if not self._built:
            self._add_internal_trainable_variables()
            self._built = True

        return rets

    def _self_attention_stack(self,
                              inputs,
                              memory,
                              decoder_self_attention_bias=None,
                              memory_attention_bias=None,
                              cache=None,
                              mode=None):
        """Stacked multihead attention module.
        """

        def _layer_norm(x, scope):
            return layers.layer_normalize(x, reuse=tf.AUTO_REUSE, scope=scope)

        inputs = tf.layers.dropout(inputs,
                                   rate=self._hparams.embedding_dropout,
                                   training=is_train_mode(mode))
        if cache is not None:
            if memory is not None:
                memory_attention_bias = \
                    cache['memory_attention_bias']
        else:
            assert decoder_self_attention_bias is not None

        x = inputs
        for i in range(self._hparams.num_blocks):
            layer_name = 'layer_{}'.format(i)
            layer_cache = cache[layer_name] if cache is not None else None
            with tf.variable_scope(layer_name) as layer_scope:
                with tf.variable_scope("self_attention"):
                    multihead_attention = \
                        self.multihead_attentions['self_att'][i]
                    selfatt_output = multihead_attention(
                        queries=_layer_norm(x, layer_scope),
                        memory=None,
                        memory_attention_bias=decoder_self_attention_bias,
                        cache=layer_cache,
                        mode=mode,
                    )
                    x = x + tf.layers.dropout(
                        selfatt_output,
                        rate=self._hparams.residual_dropout,
                        training=is_train_mode(mode),
                    )
                if memory is not None:
                    with tf.variable_scope('encdec_attention') as \
                            encdec_attention_scope:
                        multihead_attention = \
                            self.multihead_attentions['encdec_att'][i]
                        encdec_output = multihead_attention(
                            queries=_layer_norm(x, encdec_attention_scope),
                            memory=memory,
                            memory_attention_bias=memory_attention_bias,
                            mode=mode,
                        )
                        x = x + tf.layers.dropout(
                            encdec_output,
                            rate=self._hparams.residual_dropout,
                            training=is_train_mode(mode))
                poswise_network = self.poswise_networks[i]
                with tf.variable_scope('past_poswise_ln') as \
                        past_poswise_ln_scope:
                    sub_output = tf.layers.dropout(
                        poswise_network(_layer_norm(x, past_poswise_ln_scope)),
                        rate=self._hparams.residual_dropout,
                        training=is_train_mode(mode),
                    )
                    x = x + sub_output

        return _layer_norm(x, scope=self.variable_scope)

    def _init_cache(self, memory, memory_attention_bias, beam_search_decoding,
                    batch_size=None):
        """Returns an initialized cache.

        In order to support both inference-like decoding and beam-search
        decoding, the elements of each layer must be initialized and extended
        as different structure respectively. Specifically, when inference-like
        decoding, tf.TensorArray is used, which satisfies the shape consistency
        check in the while-loop in tf.contrib.seq2seq.dynamic_decode. When
        beam-search decoding, a tf.Tensor of shape
        `[batch_size, current_steps, num_units]` is maintained, where
        `current_steps` is the number of steps currently decoded.
        """
        if batch_size is None:
            batch_size = self.batch_size

        def _shape(batch_size, from_shape):
            if (not isinstance(from_shape, tf.TensorShape) or
                    from_shape.ndims == 0):
                return tf.TensorShape(None)

            batch_size = tf.contrib.util.constant_value(
                tf.convert_to_tensor(
                    batch_size, name="batch_size"))
            return tf.TensorShape([batch_size]).concatenate(from_shape)

        def _create_ta(s, d):
            return tf.TensorArray(
                dtype=d,
                size=0,
                dynamic_size=True,
                clear_after_read=False,
                element_shape=_shape(batch_size, s))

        def _create_empty_tensor(s, d):
            return tf.zeros(
                [batch_size, 0] + s.as_list(),
                dtype=d)

        _create_fn = _create_empty_tensor if beam_search_decoding else \
            _create_ta

        s = tf.TensorShape([self._hparams.multihead_attention.num_units])

        if memory is not None:
            cache = {
                'memory': memory,
                'memory_attention_bias': memory_attention_bias,
            }
            for l in range(self._hparams.num_blocks):
                cache['layer_{}'.format(l)] = {
                    'self_keys': _create_fn(s, tf.float32),
                    'self_values': _create_fn(s, tf.float32),
                    'memory_keys': _create_fn(s, tf.float32),
                    'memory_values': _create_fn(s, tf.float32),
                }
        else:
            cache = {}
            for l in range(self._hparams.num_blocks):
                cache['layer_{}'.format(l)] = {
                    'self_keys': _create_fn(s, tf.float32),
                    'self_values': _create_fn(s, tf.float32),
                }

        return cache

    def _beam_decode(self,
                     start_tokens,
                     end_token,
                     decode_length,
                     beam_width,
                     length_penalty):
        def _symbols_to_logits_fn(ids, step, cache):
            return self._input_ids_to_outputs(
                ids[:, -1], step, cache)

        outputs, log_prob = beam_search.beam_search(
            _symbols_to_logits_fn,
            start_tokens,
            beam_width,
            decode_length,
            self._vocab_size,
            length_penalty,
            eos_id=end_token,
            states=self._cache)

        # Ignores <BOS>
        outputs = outputs[:, :, 1:]
        # shape = [batch_size, seq_length, beam_width]
        outputs = tf.transpose(outputs, [0, 2, 1])
        return (outputs, log_prob)

    @property
    def batch_size(self):
        return self._helper.batch_size

    @property
    def output_size(self):
        """Output size of one step.
        """
        return TransformerDecoderOutput(
            logits=tf.TensorShape([self._vocab_size]),
            sample_id=self._helper.sample_ids_shape)

    @property
    def output_dtype(self):
        """Types of output of one step.
        """
        return TransformerDecoderOutput(
            logits=tf.float32,
            sample_id=self._helper.sample_ids_dtype)

    def initialize(self, name=None):
        """Called before any decoding iterations.

        This methods computes initial input values and initial state
        (i.e. cache).

        Args:
            name: Name scope for any created operations.

        Returns:
            `(finished, initial_inputs, initial_state)`, representing
            initial values of `finished` flags, inputs and state (i.e. cache).
        """
        return self._helper.initialize() + (self._cache,)

    def step(self, time, inputs, state, name=None):
        """Called per step of decoding.

        Args:
            time: Scalar `int32` tensor. Current step number.
            inputs: Input tensor for this time step.
            state: State (i.e. cache) from previous time step.
            name: Name scope for any created operations.

        Returns:
            `(outputs, next_state, next_inputs, finished)`. `outputs` is an
            object containing the decoder output, `next_state` is the state
            (i.e. cache), `next_inputs` is the tensor that should be used
            as input for the next step, `finished` is a boolean tensor telling
            whether the sequence is complete, for each sequence in the batch.
        """

        outputs, state = self._inputs_to_outputs(inputs, state)
        sample_ids = self._helper.sample(
            time=time, outputs=outputs, state=state)
        if self.context is not None:
            _times = tf.ones([self.batch_size], dtype=tf.int32) * time
            sample_ids = tf.where(
                self.context_sequence_length > _times,
                self.context[:, time],
                sample_ids
            )
        reach_max_time = tf.equal(time+1, self.max_decoding_length)
        finished, next_inputs, next_state = self._helper.next_inputs(
            time=time,
            outputs=outputs,
            state=state,
            sample_ids=sample_ids,
            reach_max_time=reach_max_time)
        outputs = TransformerDecoderOutput(
            logits=outputs,
            sample_id=sample_ids)
        return outputs, next_state, next_inputs, finished

    def finalize(self, outputs, final_state, sequence_lengths):
        return outputs, final_state

    @property
    def vocab_size(self):
        """The vocab size.
        """
        return self._vocab_size
