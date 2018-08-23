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
Transformer decoder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=no-name-in-module, too-many-arguments, too-many-locals

import collections

import tensorflow as tf
from tensorflow.python.framework import tensor_shape, dtypes
from tensorflow.python.util import nest

from texar.core import layers
from texar.module_base import ModuleBase
from texar.modules.networks.networks import FeedForwardNetwork
from texar.modules.embedders.position_embedders import SinusoidsPositionEmbedder
from texar.utils import beam_search
from texar.utils.shapes import shape_list, mask_sequences
from texar.utils import transformer_attentions as attentions
from texar.utils.mode import is_train_mode, is_train_mode_py

class TransformerDecoderOutput(
        collections.namedtuple("TransformerDecoderOutput",\
            ("logits", "sample_id"))):
    """The output :class:`TransformerDecoder`.

    Attributes:
        logits: A float Tensor of shape
            `[batch_size, max_time, vocab_size]` containing the logits.
        sample_id: An int Tensor of shape `[batch_size, max_time]`
            containing the sampled token indexes.
    """
    pass

class TransformerDecoder(ModuleBase):
    """Transformer decoder that applies multi-head self attention for
    sequence decoding.

    Args:
        embedding: A Tensor of shape `[vocab_size, dim]` containing the
            word embeddng. The Tensor is used as the decoder output layer.
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparamerter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter sturcture and
            default values.
    """
    def __init__(self, embedding, hparams=None):
        ModuleBase.__init__(self, hparams)

        with tf.variable_scope(self.variable_scope):
            if self._hparams.initializer:
                tf.get_variable_scope().set_initializer( \
                    layers.get_initializer(self._hparams.initializer))

            self.position_embedder = \
                SinusoidsPositionEmbedder(
                    self._hparams.position_embedder_hparams)

            self._embedding = embedding
            self._vocab_size = self._embedding.get_shape().as_list()[0]

        self.output_layer = \
            self._build_output_layer(shape_list(self._embedding)[-1])

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                "initializer": None,
                "position_embedder_hparams": None,
                "share_embed_and_transform": True,
                "transform_with_bias": True,
                "num_heads":8,
                "num_blocks":6,
                "maximum_decode_length":256,
                "embedding_dropout":0.1,
                "attention_dropout":0.1,
                "residual_dropout":0.1,
                'poswise_feedforward': {
                    'name':'ffn',
                    'layers':[
                        {
                            'type':'Dense',
                            'kwargs': {
                                'name':'conv1',
                                'units':2048,
                                'activation':'relu',
                                'use_bias':True,
                            }
                        },
                        {
                            'type':'Dropout',
                            'kwargs': {
                                'rate': 0.1,
                            }
                        },
                        {
                            'type':'Dense',
                            'kwargs': {
                                'name':'conv2',
                                'units':512,
                                'use_bias':True,
                                }
                        }
                    ],
                },
                "dim":512,
                "alpha":0,
                "name":"transformer_decoder",
            }

        Here:

        share_embed_and_transform: Choose whether to share the projection
            vector from hidden vector to logits and the word embeddings.
        transform_with_bias: Whether to apply an additional bias vector
            when projecting the hidden vector to logits.
        alpha: for length penalty. Refer to
            https://arxiv.org/abs/1609.08144.
        maximum_decode_length: The maximum length when decoding.
        The meaning of other parameters are similar to TransformerEncoder
        """
        return {
            'initializer': None,
            'position_embedder_hparams': None,
            'share_embed_and_transform': True,
            'transform_with_bias': True,
            "num_heads":8,
            "num_blocks":6,
            "maximum_decode_length":256,
            "embedding_dropout":0.1,
            'attention_dropout':0.1,
            'residual_dropout':0.1,
            'poswise_feedforward': {
		'name':'ffn',
		'layers':[
		    {
			'type':'Dense',
			'kwargs': {
			    'name':'conv1',
			    'units':2048,
			    'activation':'relu',
			    'use_bias':True,
			}
		    },
		    {
			'type':'Dropout',
			'kwargs': {
			    'rate': 0.1,
			}
		    },
		    {
			'type':'Dense',
			'kwargs': {
			    'name':'conv2',
			    'units':512,
			    'use_bias':True,
			    }
		    }
		],
            },
            'dim':512,
            'alpha':0,
            "name":"decoder",
        }

    def _prepare_tokens_to_embeds(self, tokens):
        """ a callable function to transform tokens into embeddings."""
        token_emb = tf.nn.embedding_lookup(self._embedding, tokens)
        return token_emb

    def _symbols_to_logits_fn(self, embedding_fn, max_length):
        """
            return a function to accept the decoded tokens and related
            decocoding status, and to return the logits of next token.
        """
        channels = shape_list(self._embedding)[-1]
        timing_signal = self.position_embedder(max_length, channels)
        """ the function is called in dynamic decoding.
            the ids should be `next_id` with the shape [batch_size,
                decoded_lenth]
            the returned logits is [batch_size, 1]
        """
        def _impl(ids, step, cache):
            ids = ids[:, -1:]
            inputs = embedding_fn(ids)
            # multiply embedding by sqrt of its dimention
            inputs *= self._embedding.shape.as_list()[-1]**0.5
            inputs += timing_signal[:, step:step+1]
            """
            Here we use the tensorflow flags to control the is_train_mode
            setting, instead of user passed
            """
            outputs = self._self_attention_stack(
                inputs,
                memory=cache['memory'],
                cache=cache,
            )
            #outputs = outputs[:, -1:, :]
            logits = self.output_layer(outputs)
            logits = tf.squeeze(logits, axis=[1])
            return logits, cache

        return _impl

    def _build(self,    # pylint: disable=arguments-differ
               memory,
               memory_sequence_length,
               memory_attention_bias,
               inputs,
               inputs_length,
               decoding_strategy,
               beam_width=1,
               start_token=None,
               end_token=None,
               mode=None):
        """
        This function is called on training generally.

        Args:
            memory: [batch_size, length, dim]
            memory_sequence_length: 1D Tensor of shape [batch_size]
                Only be used when memory_attention_bias is not provided.
            memory_attention_bias: 4D Tensor of shape
                `[batch_size, num_heads, length, dim]`. The attention bias
                set as a huge negative value if the correpsonding position is padding.
            inputs: Passed when training.
                Should be None when testing.
            inputs_length: 1D Tensor of shape [batch_size]
                Not used. Just for consistency. Should be passed when training.
            decoding_strategy:
                'train_greedy': For training with ground truth.
                'infer_greedy': For greedy decoding without ground truth.
                'infer_sample': For sampling decoding without ground truth.
            beam_width:
                if the beam_width of the decoder is larger than 1,
                use beam decoding regardless of decoding strategy if
                ground truth is not provided.
            start_token:
                The index of <BOS> token in the dictionary.
            end_token:
                The index of <EOS> token in the dictionary.
            mode: A python string (not a tensor), control the graph running
                flow of training or testing.

        Returns:
            logits: [batch_size, target_length, vocab_size]
            preds: [batch_size, target_length]
        """
        if memory_attention_bias is None:
            if memory_sequence_length is None:
                raise ValueError
            enc_padding = 1 - mask_sequences(tf.ones_like(memory),
                                             memory_sequence_length,
                                             tensor_rank=3)[:, :, 0]
            memory_attention_bias = attentions.attention_bias_ignore_padding(
                enc_padding)
        if decoding_strategy == 'train_greedy':
            """
            Mask the attention on future positions
            """
            decoder_self_attention_bias = (
                attentions.attention_bias_lower_triangle(
                    shape_list(inputs)[1]))
            target_inputs = inputs * self._hparams.dim**0.5

            _, lengths, channels= shape_list(target_inputs)
            pos_embeds = self.position_embedder(lengths, channels)

            inputs = target_inputs + pos_embeds

            decoder_output = self._self_attention_stack(
                inputs,
                memory,
                decoder_self_attention_bias=decoder_self_attention_bias,
                memory_attention_bias=memory_attention_bias,
                cache=None,
                mode=mode,
            )
            logits = self.output_layer(decoder_output)
            preds = tf.to_int32(tf.argmax(logits, axis=-1))
            output_length = inputs_length
            output = TransformerDecoderOutput(
                logits=logits,
                sample_id=preds
            )
            if not self._built:
                self._add_internal_trainable_variables()
                self._built = True
            return output, output_length
        else:
            # Decoding when ground truth is not provided
            batch_size = tf.shape(memory_attention_bias)[0]
            maximum_decode_length = self.hparams.maximum_decode_length
            start_tokens = tf.fill([batch_size], start_token)
            if beam_width <= 1:
                logits, preds, seq_len = self._infer_sampling(
                    self._prepare_tokens_to_embeds,
                    start_tokens,
                    end_token,
                    decode_length=maximum_decode_length,
                    memory=memory,
                    memory_attention_bias=memory_attention_bias,
                    decoding_strategy=decoding_strategy,
                )
                output = TransformerDecoderOutput(
                    logits=logits,
                    sample_id=preds
                )
                if not self._built:
                    self._add_internal_trainable_variables()
                    self._built = True
                return output, seq_len
            else:
                # The output format is different when running beam search
                sampled_ids, log_probs = self._beam_decode(
                    self._prepare_tokens_to_embeds,
                    start_tokens,
                    end_token,
                    beam_width=beam_width,
                    decode_length=maximum_decode_length,
                    memory=memory,
                    memory_attention_bias=memory_attention_bias,
                )
                predictions = {
                    'sampled_ids':sampled_ids,
                    'log_probs': log_probs
                }
                if not self._built:
                    self._add_internal_trainable_variables()
                    self._built = True
                return predictions

    def _self_attention_stack(self,
                              inputs,
                              memory,
                              decoder_self_attention_bias=None,
                              memory_attention_bias=None,
                              cache=None,
                              mode=None):
        """
            stacked multihead attention module.
        """
        inputs = tf.layers.dropout(inputs,
                                   rate=self._hparams.embedding_dropout,
                                   training=is_train_mode(mode))
        if cache is not None:
            memory_attention_bias = \
                cache['memory_attention_bias']
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
                        num_units=self._hparams.dim,
                        num_heads=self._hparams.num_heads,
                        dropout_rate=self._hparams.attention_dropout,
                        cache=layer_cache,
                        scope="multihead_attention",
                    )
                    x = x + tf.layers.dropout(
                        selfatt_output,
                        rate=self._hparams.residual_dropout,
                        training=is_train_mode(mode),
                    )
                if memory is not None:
                    with tf.variable_scope('encdec_attention'):
                        encdec_output = attentions.multihead_attention(
                            queries=layers.layer_normalize(x),
                            memory=memory,
                            memory_attention_bias=memory_attention_bias,
                            num_units=self._hparams.dim,
                            num_heads=self._hparams.num_heads,
                            dropout_rate=self._hparams.attention_dropout,
                            scope="multihead_attention"
                        )
                        x = x + tf.layers.dropout(encdec_output, \
                            rate=self._hparams.residual_dropout, \
                            training=is_train_mode(mode))
                poswise_network = FeedForwardNetwork( \
                    hparams=self._hparams['poswise_feedforward'])
                with tf.variable_scope(poswise_network.variable_scope):
                    sub_output = tf.layers.dropout(
                        poswise_network(layers.layer_normalize(x)),
                        rate=self._hparams.residual_dropout,
                        training=is_train_mode(mode),
                    )
                    x = x + sub_output

        return layers.layer_normalize(x)

    def _build_output_layer(self, dim):
        if self._hparams.share_embed_and_transform:
            if self._hparams.transform_with_bias:
                with tf.variable_scope(self.variable_scope):
                    affine_bias = tf.get_variable('affine_bias',
                                                  [self._vocab_size])
            else:
                affine_bias = None
            def outputs_to_logits(outputs):
                shape = shape_list(outputs)
                outputs = tf.reshape(outputs, [-1, dim])
                logits = tf.matmul(outputs, self._embedding, transpose_b=True)
                if affine_bias is not None:
                    logits += affine_bias
                logits = tf.reshape(logits, shape[:-1] + [self._vocab_size])
                return logits
            return outputs_to_logits
        else:
            layer = tf.layers.Dense(self._vocab_size, \
                use_bias=self._hparams.transform_with_bias)
            layer.build([None, dim])
            return layer

    @property
    def output_size(self):
        """
        The output of the _build function, (logits, preds)
        logits: [batch_size, length, vocab_size]
        preds: [batch_size, length]
        """
        return TransformerDecoderOutput(
            logits=tensor_shape.TensorShape([None, None,
                                                    self._vocab_size]),
            sample_id=tensor_shape.TensorShape([None, None])
        )

    def output_dtype(self):
        """
        The output dtype of the _build function, (float32, int32)
        """
        return TransformerDecoderOutput(
            logits=dtypes.float32, sample_id=dtypes.int32)

    def _init_cache(self, memory, memory_attention_bias):
        cache = {
            'memory': memory,
            'memory_attention_bias': memory_attention_bias,
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

    def _infer_sampling(self,
                        embedding_fn,
                        start_tokens,
                        end_token,
                        decode_length,
                        memory,
                        memory_attention_bias,
                        decoding_strategy):
        batch_size = tf.shape(start_tokens)[0]
        finished = tf.fill([batch_size], False)
        seq_length = tf.zeros([batch_size], dtype=tf.int32)
        step = tf.constant(0)
        decoded_ids = tf.zeros([batch_size, 0], dtype=tf.int32)
        logits_list = tf.zeros([batch_size, 0, self._vocab_size], dtype=tf.float32)
        next_id = tf.expand_dims(start_tokens, 1)

        cache = self._init_cache(memory, memory_attention_bias)
        symbols_to_logits_fn = self._symbols_to_logits_fn(
            embedding_fn,
            max_length=decode_length+1
        )
        def _body(step, finished, next_id, decoded_ids, cache, logits_list, seq_length):

            logits, cache = symbols_to_logits_fn(next_id, step, cache)

            if decoding_strategy == 'infer_greedy':
                next_id = tf.argmax(logits, -1, output_type=tf.int32)
            elif decoding_strategy == 'infer_sample':
                sample_id_sampler = categorical.Categorical(logis=logits)
                next_id = sample_id_sampler.sample()
                print('next id:shape:{}'.format(next_id.shape))
            cur_finished = tf.equal(next_id, end_token)
            update_len = tf.logical_and(
                tf.logical_not(finished),
                cur_finished)
            seq_length = tf.where(
                update_len,
                tf.fill(tf.shape(seq_length), step+1),
                seq_length)
            next_id = tf.expand_dims(next_id, axis=1)
            finished |= cur_finished
            #keep the shape as [batch_size, seq_len]
            logits = tf.expand_dims(logits, axis=1)
            logits_list = tf.concat([logits_list, logits], axis=1)
            decoded_ids = tf.concat([decoded_ids, next_id], axis=1)
            return step+1, finished, next_id, decoded_ids, cache, logits_list, seq_length

        def is_not_finished(i, finished, *_):
            return (i < decode_length) & tf.logical_not(tf.reduce_all(finished))

        _, _, _, decoded_ids, _, logits_list, seq_length = tf.while_loop(
            is_not_finished,
            _body,
            loop_vars=(step, finished, next_id, decoded_ids, cache, logits_list, seq_length),
            shape_invariants=(
                tf.TensorShape([]),
                tf.TensorShape([None]),
                tf.TensorShape([None, None]),
                tf.TensorShape([None, None]),
                nest.map_structure(beam_search.get_state_shape_invariants,
                                   cache),
                tf.TensorShape([None, None, None]),
                tf.TensorShape([None])
            ))

        return logits_list, decoded_ids, seq_length

    def _beam_decode(self,
                     embedding_fn,
                     start_tokens,
                     end_token,
                     memory,
                     memory_attention_bias,
                     decode_length=256,
                     beam_width=5):
        cache = self._init_cache(memory, memory_attention_bias)
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
            eos_id=end_token)

        outputs = outputs[:, :, 1:] # ignore <BOS>
        outputs = tf.transpose(outputs, [0, 2, 1]) #[batch_size, seq_length, beam_width]
        return (outputs, log_probs)
