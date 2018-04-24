"""
 transformer decoders. Attention is all you need.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=no-name-in-module, too-many-arguments, too-many-locals
# pylint: disable=not-context-manager

import collections

import tensorflow as tf
from tensorflow.python.framework import tensor_shape, dtypes

from texar.core import layers, attentions
from texar import context
from texar.module_base import ModuleBase
from texar.modules.networks.networks import FeedForwardNetwork
from texar.modules.embedders import embedder_utils
from texar.utils import beam_search

class TransformerDecoderOutput(
        collections.namedtuple("TransformerDecoderOutput",\
            ("output_logits", "sample_ids"))):
    """the output logits and sampled_ids"""
    pass

class TransformerDecoder(ModuleBase):
    """decoder for transformer: Attention is all you need
    """
    def __init__(self, embedding=None, vocab_size=None, hparams=None):
        ModuleBase.__init__(self, hparams)
        self._vocab_size = vocab_size
        self._embedding = None
        if self._hparams.initializer:
            with tf.variable_scope(self.variable_scope):
                tf.get_variable_scope().set_initializer(
                        layers.get_initializer(self._hparams.initializer))
        if self._hparams.use_embedding:
            if embedding is None and vocab_size is None:
                raise ValueError("If 'embedding' is not provided, 'vocab_size' must be specified.")
            if isinstance(embedding, tf.Tensor):
                self._embedding = embedding
            else:
                self._embedding = embedder_utils.get_embedding(
                    self._hparams.embedding, embedding, vocab_size,
                    variable_scope=self.variable_scope)
                self._embed_dim = layers.shape_list(self._embedding)[-1]
                if self._hparams.zero_pad:
                    self._embedding = tf.concat((tf.zeros(shape=[1, self._embed_dim]),\
                        self._embedding[1:, :]), 0)
            if self._vocab_size is None:
                self._vocab_size = self._embedding.get_shape().as_list()[0]
        self.output_layer = self.build_output_layer(layers.shape_list(self._embedding)[-1])
    @staticmethod
    def default_hparams():
        return {
            'initializer':None,
            'multiply_embedding_mode': 'sqrt_depth',
            'share_embed_and_transform': True,
            'transform_with_bias': True,
            "use_embedding": True,
            "name":"decoder",
            "num_heads":8,
            "num_blocks":6,
            "zero_pad": True,
            "max_seq_length":10,
            "maximum_decode_length":10,
            "beam_width":1,
            'alpha':0,
            "embedding_dropout":0.1,
            'attention_dropout':0.1,
            'residual_dropout':0.1,
            "sinusoid":True,
            'poswise_feedforward':None,
            'num_units':512,
        }
    def prepare_tokens_to_embeds(self, tokens):
        token_emb = tf.nn.embedding_lookup(self._embedding, tokens)
        return token_emb

    def _symbols_to_logits_fn(self, embedding_fn, max_length):
        timing_signal = layers.get_timing_signal_1d(max_length,
            self._embedding.shape.as_list()[-1])
        def _impl(ids, step, cache):
            inputs = embedding_fn(ids[:, -1:])
            if self._hparams.multiply_embedding_mode == 'sqrt_depth':
                inputs *= self._embedding.shape.as_list()[-1]**0.5
            inputs += timing_signal[:, step:step+1]
            outputs = self._self_attention_stack(
                inputs,
                encoder_output=cache['memory'],
                cache=cache,
            )
            outputs = outputs[:, -1:, :]
            logits = self.output_layer(outputs)
            return logits, cache
        return _impl

    def _build(self, targets, encoder_output, encoder_decoder_attention_bias):
        """
            this function is called on training generally.
            Args:
                targets: [bath_size, target_length], begins with [bos] token
                encoder_output: [batch_size, source_length, channels]
            outputs:
                logits: [batch_size, target_length, vocab_size]
                preds: [batch_size, target_length]
        """
        logits = None
        if targets is not None:
            decoder_self_attention_bias = (
                attentions.attention_bias_lower_triangle(
                    layers.shape_list(targets)[1]))
            target_inputs = tf.nn.embedding_lookup(self._embedding, targets)
            if self._hparams.multiply_embedding_mode == 'sqrt_depth':
                target_inputs = target_inputs * \
                    (self._embedding.shape.as_list()[-1]**0.5)
            logits = self.decode(
                target_inputs,
                encoder_output,
                encoder_decoder_attention_bias,
                decoder_self_attention_bias,
            )
        preds = tf.to_int32(tf.argmax(logits, axis=-1))
        return logits, preds

    def dynamic_decode(self, encoder_output, encoder_decoder_attention_bias):
        """
            this function is called on in test mode, without the target input.
            Based on beam_size, here performs greedy decoding or beam search decoding
        """
        with tf.variable_scope(self.variable_scope, reuse=True):
            batch_size = tf.shape(encoder_decoder_attention_bias)[0]
            beam_width = self._hparams.beam_width
            maximum_decode_length = self.hparams.maximum_decode_length
            start_tokens = tf.fill([batch_size], 1)
            EOS = 2
            if beam_width <= 1:
                sampled_ids, _ , sampled_length, log_probs = self.greedy_decode(
                    self.prepare_tokens_to_embeds,
                    start_tokens,
                    EOS,
                    maximum_iterations=maximum_decode_length,
                    memory=encoder_output,
                    encoder_decoder_attention_bias=encoder_decoder_attention_bias
                )
            else:
                sampled_ids, _ , sampled_length, log_probs = self.beam_decode(
                    self.prepare_tokens_to_embeds,
                    start_tokens,
                    EOS,
                    beam_width=beam_width,
                    alpha=self._hparams.alpha,
                    decode_length=maximum_decode_length,
                    memory=encoder_output,
                    encoder_decoder_attention_bias=encoder_decoder_attention_bias,
                )
            predictions = {
                'sampled_ids':sampled_ids,
                'length': sampled_length,
                'log_probs': log_probs
            }
        return predictions

    def decode(self,
               inputs,
               encoder_output,
               encoder_decoder_attention_bias,
               decoder_self_attention_bias,
               cache=None
        ):
        inputs =layers.add_timing_signal_1d(inputs)
        decoder_output = self._self_attention_stack(
            inputs,
            encoder_output,
            decoder_self_attention_bias,
            encoder_decoder_attention_bias,
            cache=cache
        )

        logits = self.output_layer(decoder_output)
        return logits


    def _self_attention_stack(self,
                              inputs,
                              encoder_output=None,
                              decoder_self_attention_bias=None,
                              encoder_decoder_attention_bias=None,
                              cache=None):
        inputs = tf.layers.dropout(inputs, rate=self._hparams.embedding_dropout,
            training=context.global_mode_train())
        if encoder_output is not None:
            if cache is not None:
                encoder_decoder_attention_bias = cache['encoder_decoder_attention_bias']

        x = inputs
        for i in range(self._hparams.num_blocks):
            layer_name = 'layer_{}'.format(i)
            layer_cache = cache[layer_name] if cache is not None else None
            with tf.variable_scope(layer_name):
                with tf.variable_scope("self_attention"):
                    selfatt_output = layers.multihead_attention(
                        queries=layers.layer_normalize(x),
                        memory=None,
                        bias=decoder_self_attention_bias,
                        num_units=self._hparams.num_units,
                        num_heads=self._hparams.num_heads,
                        dropout_rate=self._hparams.attention_dropout,
                        cache=layer_cache,
                        scope="multihead_attention",
                    )
                    x = x + tf.layers.dropout(
                        selfatt_output,
                        rate=self._hparams.residual_dropout,
                        training=context.global_mode_train()
                    )
                with tf.variable_scope('encdec_attention'):
                    encdec_output = layers.multihead_attention(
                        queries=layers.layer_normalize(x),
                        memory=encoder_output,
                        bias=encoder_decoder_attention_bias,
                        num_units=self._hparams.num_units,
                        num_heads=self._hparams.num_heads,
                        dropout_rate=self._hparams.attention_dropout,
                        causality=False,
                        scope="multihead_attention"
                    )
                    x = x + tf.layers.dropout(encdec_output, \
                        rate=self._hparams.residual_dropout,
                        training=context.global_mode_train()
                    )
                poswise_network = FeedForwardNetwork(hparams=self._hparams['poswise_feedforward'])
                with tf.variable_scope(poswise_network.variable_scope):
                    sub_output = tf.layers.dropout(
                        poswise_network(layers.layer_normalize(x)),
                        rate=self._hparams.residual_dropout,
                        training=context.global_mode_train()
                    )
                    x = x + sub_output

        return layers.layer_normalize(x)

    def build_output_layer(self, num_units):
        if self._hparams.share_embed_and_transform:
            with tf.variable_scope(self.variable_scope):
                affine_bias = tf.get_variable('affine_bias',
                    [self._vocab_size])
            affine_bias = None
            def outputs_to_logits(outputs):
                shape = layers.shape_list(outputs)
                outputs = tf.reshape(outputs, [-1, num_units])
                logits = tf.matmul(outputs, self._embedding, transpose_b=True)
                if affine_bias:
                    logits += affine_bias
                logits = tf.reshape(logits, shape[:-1] + [self._vocab_size])
                return logits
            return outputs_to_logits
        else:
            layer = tf.layers.Dense(self._vocab_size, \
                use_bias=self._hparams.transform_with_bias)
            layer.build([None, num_units])
            return layer

    @property
    def output_size(self):
        """
        The output of the _build function, (logits, preds)
        logits: [batch_size, length, vocab_size]
        preds: [batch_size, length]
        """
        return TransformerDecoderOutput(
            output_logits=tensor_shape.TensorShape([None, None, self._vocab_size]),
            sample_id=tensor_shape.TensorShape([None, None])
            )

    def output_dtype(self):
        """
        The output dtype of the _build function, (float32, int32)
        """
        return TransformerDecoderOutput(
            output_logits=dtypes.float32, sample_id=dtypes.int32)
    def _init_cache(self, memory, encoder_decoder_attention_bias):
        cache = {
            'memory': memory,
            'encoder_decoder_attention_bias': encoder_decoder_attention_bias,
        }
        batch_size = tf.shape(memory)[0]
        depth = memory.get_shape().as_list()[-1]
        for l in range(self._hparams.num_blocks):
            cache['layer_{}'.format(l)] = {
                'self_keys': tf.zeros([batch_size, 0, depth]),
                'self_values': tf.zeros([batch_size, 0, depth]),
                'memory_keys': tf.zeros([batch_size, 0, depth]),
                'memory_values': tf.zeros([batch_size, 0, depth]),
            }
        return cache

    def greedy_decode(self,
                      embedding_fn,
                      start_tokens,
                      EOS,
                      decode_length,
                      memory,
                      encoder_decoder_attention_bias):
        batch_size = tf.shape(start_tokens)[0]
        finished = tf.tile([False], [batch_size])
        step = tf.constant(0)
        inputs = tf.expand_dims(start_tokens, 1)
        lengths = tf.zeros([batch_size], dtype=tf.int32)
        log_probs = tf.zeros([batch_size])
        cache = self._init_cache(memory, encoder_decoder_attention_bias)
        symbols_to_logits_fn = self._symbols_to_logits_fn(embedding_fn,
            maximum_length=decode_length+1)
        def _body(step, finished, inputs, lengths, log_probs, cache):
            inputs_lengths = tf.add(lengths, 1 - tf.cast(finished, lengths.dtype))
            logits, cache = symbols_to_logits_fn(inputs, step, cache)
            log_probs = tf.nn.log_softmax(logits)
            sample_ids = tf.argmax(log_probs, axis=-1)

            sample_probs = tf.reduce_max(log_probs, axis=-1)
            masked_probs = tf.squeeze(sample_probs, -1) * \
                (1.0 - tf.cast(finished, sample_probs.dtype))
            log_probs = tf.add(log_probs, masked_probs)

            next_inputs = tf.concat([inputs, tf.cast(sample_ids, inputs.dtype)], -1)
            next_lengths = inputs_lengths
            finished = tf.logical_or(
                finished,
                tf.equal(tf.squeeze(sample_ids, axis=[-1]), EOS)
            )
            return step+1, finished, next_inputs, next_lengths, log_probs, cache

        def is_not_finished(i, finished, *_):
            return (i < decode_length) & tf.logical_not(tf.reduce_all(finished))

        step, _, outputs, lengths, log_probs, _ = tf.while_loop(
            is_not_finished,
            _body,
            loop_vars=(step, finished, inputs, lengths, log_probs, cache),
            shape_invariants=(
                tf.TensorShape([]),
                finished.get_shape(),
                tf.TensorShape([None, None]),
                lengths.get_shape(),
                log_probs.get_shape(),
                tf.contrib.framework.nest.map_structure(beam_search.get_state_shape_invariants, cache)
            ),
            parallel_iterations=1)

        outputs = tf.slice(outputs, [0, 1], [-1, -1]) #[ignore <s>

        outputs = tf.expand_dims(outputs, 1)
        lengths = tf.expand_dims(lengths, 1)
        log_probs = tf.expand_dims(log_probs, 1)

        return (outputs, None, lengths, log_probs)

    def beam_decode(self,
                    embedding_fn,
                    start_tokens,
                    EOS,
                    memory=None,
                    encoder_decoder_attention_bias=None,
                    alpha=0.0,
                    decode_length=256,
                    beam_width=5):
        cache = self._init_cache(memory, encoder_decoder_attention_bias)
        symbols_to_logits_fn = self._symbols_to_logits_fn(embedding_fn,
            max_length=decode_length+1)
        outputs, log_probs = beam_search.beam_search(
            symbols_to_logits_fn,
            start_tokens,
            beam_width,
            decode_length,
            self._vocab_size,
            alpha,
            states=cache,
            eos_id=EOS)

        outputs = tf.slice(outputs, [0, 0, 1], [-1, -1, -1]) # Ignore <s>.

        lengths = tf.reduce_sum(tf.cast(tf.not_equal(outputs, 0), tf.int32), axis=-1)

        return (outputs, None, lengths, log_probs)
