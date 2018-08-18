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
from tensorflow.python.util import nest

from texar.core import layers
from texar.module_base import ModuleBase
from texar.modules.networks.networks import FeedForwardNetwork
from texar.modules.embedders import embedder_utils
from texar.modules.embedders import position_embedders
from texar import utils
from texar.utils import beam_search
from texar.utils.shapes import shape_list
from texar.utils import transformer_attentions as attentions

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
        with tf.variable_scope(self.variable_scope):
            if self._hparams.initializer:
                tf.get_variable_scope().set_initializer( \
                    layers.get_initializer(self._hparams.initializer))
            if self._hparams.position_embedder.name == 'sinusoids':
                self.position_embedder = \
                    position_embedders.SinusoidsPositionEmbedder( \
                    self._hparams.position_embedder.hparams)

        if self._hparams.use_embedding:
            if embedding is None and vocab_size is None:
                raise ValueError("""If 'embedding' is not provided,
                    'vocab_size' must be specified.""")
            if isinstance(embedding, (tf.Tensor, tf.Variable)):
                self._embedding = embedding
            else:
                self._embedding = embedder_utils.get_embedding(
                    self._hparams.embedding, embedding, vocab_size,
                    variable_scope=self.variable_scope)
                self._embed_dim = shape_list(self._embedding)[-1]
                if self._hparams.zero_pad:
                    self._embedding = tf.concat( \
                        (tf.zeros(shape=[1, self._embed_dim]),\
                        self._embedding[1:, :]), 0)
            if self._vocab_size is None:
                self._vocab_size = self._embedding.get_shape().as_list()[0]
        self.output_layer = \
            self.build_output_layer(shape_list(self._embedding)[-1])
    @staticmethod
    def default_hparams():
        """default hyperrams for transformer deocder.
            sampling_method: argmax or sample. To choose the function transforming the logits to the sampled id in the next position when inferencing.
        """
        return {
            'sampling_method': 'argmax',
            'initializer': None,
            'multiply_embedding_mode': 'sqrt_depth',
            'position_embedder': None,
            'share_embed_and_transform': True,
            'transform_with_bias': True,
            "use_embedding": True,
            "name":"decoder",
            "num_heads":8,
            "num_blocks":6,
            "zero_pad": False,
            "bos_pad": False,
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
            'eos_idx': 2,
            'bos_idx': 1,
        }

    def prepare_tokens_to_embeds(self, tokens):
        """ a callable function to transform tokens into embeddings."""
        token_emb = tf.nn.embedding_lookup(self._embedding, tokens)
        return token_emb

    def _symbols_to_logits_fn(self, embedding_fn, max_length):
        channels = shape_list(self._embedding)[-1]
        timing_signal = self.position_embedder(max_length, channels)

        """ the function is normally called in dynamic decoding mode.
                the ids should be `next_id` with the shape [batch_size, 1]
            the returned logits is [batch_size, 1]
        """
        def _impl(ids, step, cache):
            ids = ids[:, -1:]
            inputs = embedding_fn(ids)
            if self._hparams.multiply_embedding_mode == 'sqrt_depth':
                inputs *= self._embedding.shape.as_list()[-1]**0.5
            else:
                assert NotImplementedError
            inputs += timing_signal[:, step:step+1]

            outputs = self._self_attention_stack(
                inputs,
                encoder_output=cache['memory'],
                cache=cache,
            )
            #outputs = outputs[:, -1:, :]
            logits = self.output_layer(outputs)
            logits = tf.squeeze(logits, axis=[1])

            return logits, cache

        return _impl
    #pylint:disable=arguments-differ
    def _build(self, decoder_input, encoder_output, \
        encoder_decoder_attention_bias, mode=None):
        """
            this function is called on training generally.
            Args:
                targets: [bath_size, target_length], generally begins with [bos] token
                encoder_output: [batch_size, source_length, channels]
            outputs:
                logits: [batch_size, target_length, vocab_size]
                preds: [batch_size, target_length]
        """
        logits = None
        decoder_self_attention_bias = (
            attentions.attention_bias_lower_triangle(
                shape_list(decoder_input)[1]))
        target_inputs = tf.nn.embedding_lookup(self._embedding, decoder_input)
        if self._hparams.multiply_embedding_mode == 'sqrt_depth':
            target_inputs = target_inputs * \
                (self._embedding.shape.as_list()[-1]**0.5)
        lengths = shape_list(target_inputs)[1]
        channels = shape_list(target_inputs)[2]
        pos_embeds = self.position_embedder(lengths, channels)
        inputs = target_inputs + pos_embeds
        self.decoder_output = self._self_attention_stack(
            inputs,
            encoder_output,
            decoder_self_attention_bias=decoder_self_attention_bias,
            encoder_decoder_attention_bias=encoder_decoder_attention_bias,
            cache=None,
            mode=None,
        )

        logits = self.output_layer(self.decoder_output)
        preds = tf.to_int32(tf.argmax(logits, axis=-1))

        if not self._built:
            self._add_internal_trainable_variables()
            self._built = True

        return logits, preds

    def dynamic_decode(self, encoder_output, encoder_decoder_attention_bias):
        """
            this function is called on in test mode, without the target input.
        """
        with tf.variable_scope(self.variable_scope, reuse=True):
            batch_size = tf.shape(encoder_decoder_attention_bias)[0]
            beam_width = self._hparams.beam_width
            maximum_decode_length = self.hparams.maximum_decode_length
            start_tokens = tf.fill([batch_size], 1)
            if beam_width <= 1:
                sampled_ids, log_probs = self.greedy_decode(
                    self.prepare_tokens_to_embeds,
                    start_tokens,
                    self._hparams.eos_idx,
                    decode_length=maximum_decode_length,
                    memory=encoder_output,
                    encoder_decoder_attention_bias=\
                        encoder_decoder_attention_bias
                )
            else:
                sampled_ids, log_probs = self.beam_decode(
                    self.prepare_tokens_to_embeds,
                    start_tokens,
                    self._hparams.eos_idx,
                    beam_width=beam_width,
                    decode_length=maximum_decode_length,
                    memory=encoder_output,
                    encoder_decoder_attention_bias=\
                        encoder_decoder_attention_bias,
                )
            predictions = {
                'sampled_ids':sampled_ids,
                'log_probs': log_probs
            }
        return predictions

    def _self_attention_stack(self,
                              inputs,
                              encoder_output,
                              decoder_self_attention_bias=None,
                              encoder_decoder_attention_bias=None,
                              cache=None,
                              mode=None):
        """
            stacked multihead attention module.
        """
        inputs = tf.layers.dropout(inputs,
                                   rate=self._hparams.embedding_dropout,
                                   training=utils.mode.is_train_mode(mode))
        if cache is not None:
            encoder_decoder_attention_bias = \
                cache['encoder_decoder_attention_bias']
        else:
            assert decoder_self_attention_bias is not None

        x = inputs
        for i in range(self._hparams.num_blocks):
            layer_name = 'layer_{}'.format(i)
            layer_cache = cache[layer_name] if cache is not None else None
            with tf.variable_scope(layer_name):
                with tf.variable_scope("self_attention"):
                    selfatt_output = attentions.multihead_attention(
                        queries=layers.layer_normalize(x),
                        memory=None,
                        memory_attention_bias=decoder_self_attention_bias,
                        num_units=self._hparams.num_units,
                        num_heads=self._hparams.num_heads,
                        dropout_rate=self._hparams.attention_dropout,
                        cache=layer_cache,
                        scope="multihead_attention",
                    )
                    x = x + tf.layers.dropout(
                        selfatt_output,
                        rate=self._hparams.residual_dropout,
                        training=utils.mode.is_train_mode(mode),
                    )
                if encoder_output is not None:
                    with tf.variable_scope('encdec_attention'):
                        encdec_output = attentions.multihead_attention(
                            queries=layers.layer_normalize(x),
                            memory=encoder_output,
                            memory_attention_bias=encoder_decoder_attention_bias,
                            num_units=self._hparams.num_units,
                            num_heads=self._hparams.num_heads,
                            dropout_rate=self._hparams.attention_dropout,
                            scope="multihead_attention"
                        )
                        x = x + tf.layers.dropout(encdec_output, \
                            rate=self._hparams.residual_dropout, \
                            training=utils.mode.is_train_mode(mode),
                        )
                poswise_network = FeedForwardNetwork( \
                    hparams=self._hparams['poswise_feedforward'])
                with tf.variable_scope(poswise_network.variable_scope):
                    sub_output = tf.layers.dropout(
                        poswise_network(layers.layer_normalize(x)),
                        rate=self._hparams.residual_dropout,
                        training=utils.mode.is_train_mode(mode),
                    )
                    x = x + sub_output

        return layers.layer_normalize(x)

    def build_output_layer(self, num_units):
        if self._hparams.share_embed_and_transform:
            if self._hparams.transform_with_bias:
                with tf.variable_scope(self.variable_scope):
                    affine_bias = tf.get_variable('affine_bias',
                        [self._vocab_size])
            else:
                affine_bias = None
            def outputs_to_logits(outputs):
                shape = shape_list(outputs)
                outputs = tf.reshape(outputs, [-1, num_units])
                logits = tf.matmul(outputs, self._embedding, transpose_b=True)
                if affine_bias is not None:
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
        finished = tf.fill([batch_size], False)
        step = tf.constant(0)
        decoded_ids = tf.zeros([batch_size, 0], dtype=tf.int32)
        next_id = tf.expand_dims(start_tokens, 1)
        print('next id:{}'.format(next_id.shape))
        log_prob = tf.zeros([batch_size], dtype=tf.float32)

        cache = self._init_cache(memory, encoder_decoder_attention_bias)
        symbols_to_logits_fn = self._symbols_to_logits_fn(embedding_fn,
            max_length=decode_length+1)

        def _body(step, finished, next_id, decoded_ids, cache, log_prob):

            logits, cache = symbols_to_logits_fn(next_id, step, cache)
            log_probs = logits - \
                tf.reduce_logsumexp(logits, axis=-1, keep_dims=True)

            #TODO: by default, the output_type is tf.int64.
            # Can we adjust the default int type of texar to tf.int64?
            if self.sampling_method == 'argmax':
                next_id = tf.argmax(logits, -1, output_type=tf.int32)
            elif self.sampling_method == 'sample':
                next_id = tf.multinomial(logits, 1).squeeze(axis=1)
            finished |= tf.equal(next_id, EOS)
            log_prob_indices = tf.stack(
                [tf.range(tf.to_int32(batch_size)), next_id], axis=1)
            log_prob += tf.gather_nd(log_probs, log_prob_indices)

            next_id = tf.expand_dims(next_id, axis=1)
            #keep the shape as [batch_size, seq_len]

            decoded_ids = tf.concat([decoded_ids, next_id], axis=1)
            return step+1, finished, next_id, decoded_ids, cache, log_prob

        def is_not_finished(i, finished, *_):
            return (i < decode_length) & tf.logical_not(tf.reduce_all(finished))

        _, _, _, decoded_ids, _, log_prob = tf.while_loop(
            is_not_finished,
            _body,
            loop_vars=(step, finished, next_id, decoded_ids, cache, log_prob),
            shape_invariants=(
                tf.TensorShape([]),
                tf.TensorShape([None]),
                tf.TensorShape([None, None]),
                tf.TensorShape([None, None]),
                nest.map_structure(beam_search.get_state_shape_invariants, cache),
                tf.TensorShape([None]),
            ))

        outputs = tf.expand_dims(decoded_ids, 1)
        log_prob = tf.expand_dims(log_prob, 1)
        return (outputs, log_prob)

    def beam_decode(self,
                    embedding_fn,
                    start_tokens,
                    EOS,
                    memory,
                    encoder_decoder_attention_bias,
                    decode_length=256,
                    beam_width=5):
        cache = self._init_cache(memory, encoder_decoder_attention_bias)
        symbols_to_logits_fn = self._symbols_to_logits_fn(embedding_fn, \
            max_length=decode_length+1)
        outputs, log_probs = beam_search.beam_search(
            symbols_to_logits_fn,
            start_tokens,
            beam_width,
            decode_length,
            self._vocab_size,
            self._hparams.alpha,
            states=cache,
            eos_id=EOS)

        outputs = outputs[:, :, 1:] # ignore <BOS>
        return (outputs, log_probs)
