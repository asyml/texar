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
XLNet encoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from texar.core import layers

from texar.hyperparams import HParams

from texar.modules.xlnet import model_utils
from texar.modules.berts import bert_utils
from texar.modules.embedders import WordEmbedder
from texar.modules.encoders import EncoderBase

from texar.module_base import ModuleBase

from texar.utils import dict_fetch

__all__ = [
    "XLNetEncoder",
    "PositionWiseFF",
    "RelativePositionalEncoding",
    "RelativeMutiheadAttention"
]


class PositionWiseFF(ModuleBase):
    def __init__(self, hparams=None):
        super().__init__(hparams)

        hidden_dim = self._hparams.hidden_dim
        ffn_inner_dim = self._hparams.ffn_inner_dim
        dropout = self._hparams.dropout
        activation = self._hparams.activation
        if activation == 'gelu':
            activation = layers.gelu

        l1_hparams = {
            "type": "Dense",
            "kwargs": {
                "units": ffn_inner_dim,
                "activation": activation
            }
        }
        self.linear1 = layers.get_layer(hparams=l1_hparams)
        dropout_hparams = {
            "type": "Dropout",
            "kwargs": {
                "rate": dropout
            }
        }
        self.dropout = layers.get_layer(hparams=dropout_hparams)
        l2_hparams = {
            "type": "Dense",
            "kwargs": {
                "units": hidden_dim,
                "activation": activation
            }
        }
        self.linear2 = layers.get_layer(hparams=l2_hparams)

    @staticmethod
    def default_hparams():
        return {
            "name": "position_wise_FF",
            "hidden_dim": 1024,
            "ffn_inner_dim": 4096,
            "dropout": 0.1,
            "activation": 'gelu',
        }

    def _build(self, input):
        # Position-wise feed-forward
        output = self.linear1(input)
        output = self.dropout(output)
        output = self.linear2(output)
        output = self.dropout(output)
        # residual + layer norm
        output = tf.contrib.layers.layer_norm(
            input + output, begin_norm_axis=-1, scope=self.variable_scope)
        return output


class PositionalEmbedding(ModuleBase):
    def __init__(self, embed_dim):
        super().__init__()
        freq_seq = tf.range(0.0, embed_dim, 2.0)
        self.inv_freq = 1 / (10000 ** (freq_seq / embed_dim))

    def _build(self, pos_seq):
        pos_seq = tf.dtypes.cast(pos_seq, dtype=self.inv_freq.dtype)
        sinusoid_inp = tf.einsum('i,d->id', pos_seq, self.inv_freq)
        pos_emb = tf.concat([tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)], -1)
        return pos_emb


class RelativePositionalEncoding(ModuleBase):
    def __init__(self, hparams=None):
        super().__init__(hparams)
        self.sinusoid_embed = PositionalEmbedding(self._hparams.dim)

    @staticmethod
    def default_hparams():
        return {
            "name": "relative_positional_encoder",
            "dim": 1024,
            "max_seq_len": 512,
        }

    def _create_positional_embedding(self, start, end, step, batch_size,
                                     clamp_len=None):
        pos_seq = tf.range(start, end, step)
        if clamp_len is not None:
            pos_seq = tf.clip_by_value(pos_seq, -clamp_len, clamp_len)
        pos_emb = self.sinusoid_embed(pos_seq)
        pos_emb = pos_emb[:, None, :]

        if batch_size is not None:
            pos_emb = tf.tile(pos_emb, [1, batch_size, 1])

        return pos_emb

    def _build(self, batch_size, seq_len, total_len, clamp_len=None,
               attn_type='bi', bi_data=True):
        if attn_type == 'bi':
            start, end = total_len, -seq_len
        elif attn_type == 'uni':
            start, end = total_len, -1
        else:
            raise ValueError(f"Unknown `attn_type` {attn_type}")

        if bi_data:
            if batch_size % 2 != 0:
                raise ValueError("`batch_size` must be an even number")
            fwd_pos_embed = self._create_positional_embedding(
                start, end, -1, batch_size // 2, clamp_len)
            bwd_pos_embed = self._create_positional_embedding(
                -start, -end, 1, batch_size // 2, clamp_len)
            pos_embed = tf.concat([fwd_pos_embed, bwd_pos_embed], axis=1)
        else:
            pos_embed = self._create_positional_embedding(
                start, end, -1, batch_size, clamp_len)
        return pos_embed


class RelativeMutiheadAttention(ModuleBase):
    def __init__(self, r_r_bias, r_w_bias, r_s_bias, hparams=None):
        super().__init__(hparams=hparams)

        self.num_heads = self._hparams.num_heads
        self.head_dim = self._hparams.head_dim
        hidden_dim = self._hparams.hidden_dim

        with tf.variable_scope(self.variable_scope):
            if self._hparams.initializer:
                tf.get_variable_scope().set_initializer(
                    layers.get_initializer(self._hparams.initializer))

            self.head_projection = layers.get_layer(hparams={
                "type": "Dense",
                "kwargs": {"units": 3 * self.num_heads * self.head_dim,
                           "use_bias": False}
            })

            self.pos_projection = layers.get_layer(hparams={
                "type": "Dense",
                "kwargs": {
                    "units": self.num_heads * self.head_dim,
                    "use_bias": False
                }
            })

            self.dropout = layers.get_layer(hparams={
                "type": "Dropout",
                "kwargs": {
                    "rate": self._hparams.dropout
                }
            })

            self.dropout_attn = layers.get_layer(hparams={
                "type": "Dropout",
                "kwargs": {
                    "rate": self._hparams.attention_dropout
                }
            })

            self.output_projection = layers.get_layer(hparams={
                "type": "Dense",
                "kwargs": {
                    "units": hidden_dim,
                    "use_bias": False
                }
            })

            bias_shape = (self.num_heads, self.head_dim)
            self.untie_r = r_r_bias is None
            self.r_r_bias = (r_r_bias if r_r_bias is not None else
                             tf.get_variable('r_r_bias', shape=bias_shape))
            self.r_w_bias = (r_w_bias if r_w_bias is not None else
                             tf.get_variable('r_w_bias', shape=bias_shape))

            if self._hparams.use_segments:
                self.segment_embed = tf.get_variable(
                    'segment_embedding', [2, self.num_heads, self.head_dim])
                self.r_s_bias = (r_s_bias if r_s_bias is not None else
                                 tf.get_variable('r_s_bias', shape=bias_shape))

            self.scale = 1 / (self.head_dim ** 0.5)

    @staticmethod
    def default_hparams():
        return {
            "name": "relative_multihead_attention",
            "initializer": None,
            "num_heads": 16,
            "hidden_dim": 1024,
            "head_dim": 64,
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "use_segments": True,
        }

    @staticmethod
    def _rel_shift(x, klen=-1):
        """Perform relative shift to form the relative attention score."""
        x_size = tf.shape(x)

        x = tf.reshape(x, [x_size[1], x_size[0], x_size[2], x_size[3]])
        x = tf.slice(x, [1, 0, 0, 0], [-1, -1, -1, -1])
        x = tf.reshape(x, [x_size[0], x_size[1] - 1, x_size[2], x_size[3]])
        x = tf.slice(x, [0, 0, 0, 0], [-1, klen, -1, -1])

        return x

    def _compute_attention_score(self, q_head, k_head_h, v_head_h, k_head_r,
                                 segment_mat, attn_mask=None, **kwargs):
        is_training = kwargs["is_training"]

        # Content based attention score.
        q_head_rw = q_head + self.r_w_bias
        # attn_ac: (seq_len, tot_len, batch_size, n_head)
        attn_ac = tf.einsum('ibnd,jbnd->ijbn', q_head_rw, k_head_h)

        # Position based attention score.
        q_head_rr = q_head + self.r_r_bias
        # attn_bd: (seq_len, tot_len, batch_size, n_head)
        attn_bd = tf.einsum('ibnd,jbnd->ijbn', q_head_rr, k_head_r)
        attn_bd = self._rel_shift(attn_bd, klen=tf.shape(attn_ac)[1])

        # Segment based attention score.
        if segment_mat is None:
            attn_ef = 0
        else:
            q_head_rs = q_head + self.r_s_bias
            attn_ef = tf.einsum(
                'ibnd,snd->ibns', q_head_rs, self.segment_embed)
            attn_ef = tf.einsum('ijbs,ibns->ijbn', segment_mat, attn_ef)

        # Merge attention scores and perform masking.
        # attn_score: (seq_len, tot_len, batch_size, n_head)
        attn_score = (attn_ac + attn_bd + attn_ef) * self.scale
        if attn_mask is not None:
            # attn_score = attn_score * (1 - attn_mask) - 1e30 * attn_mask
            attn_score = attn_score - 1e30 * attn_mask

        # attention probability
        attn_prob = tf.nn.softmax(attn_score, 1)
        attn_prob = self.dropout(attn_prob, training=is_training)

        # attention output
        attn_vec = tf.einsum('ijbn,jbnd->ibnd', attn_prob, v_head_h)

        return attn_vec

    def _post_attention(self, attn_vec):
        shape = tf.shape(attn_vec)
        attn_vec = tf.reshape(tensor=attn_vec, shape=[shape[0], shape[1], self.num_heads*self.head_dim])
        attn_out = self.output_projection(attn_vec)
        attn_out = self.dropout(attn_out)
        return attn_out

    def _build(self, states_h, states_g, pos_embed, segment_mat,
               attn_mask_h=None, attn_mask_g=None, target_mapping=None,
               memory=None, **kwargs):
        shape = tf.shape(states_h)
        seq_len, batch_size = shape[0], shape[1]
        pos_len = tf.shape(pos_embed)[0]

        if memory is not None and memory.dim() > 1:
            concat_input = tf.concat([memory, states_h], axis=0)
        else:
            concat_input = states_h

        # Content heads.
        heads = self.head_projection(concat_input)
        q_head_h, k_head_h, v_head_h = tf.split(heads, num_or_size_splits=3, axis=-1)
        q_head_h = q_head_h[-seq_len:]
        tot_len = tf.shape(k_head_h)[0]

        q_head_h = tf.reshape(
            tensor=q_head_h,
            shape=[seq_len, batch_size, self.num_heads, self.head_dim])
        k_head_h = tf.reshape(
            tensor=k_head_h,
            shape=[tot_len, batch_size, self.num_heads, self.head_dim])
        v_head_h = tf.reshape(
            tensor=v_head_h,
            shape=[tot_len, batch_size, self.num_heads, self.head_dim])

        # Positional heads.
        k_head_r = self.pos_projection(pos_embed)
        k_head_r = tf.reshape(
            tensor=k_head_r,
            shape=[pos_len, batch_size, self.num_heads, self.head_dim])

        # Core attention ops.
        attn_vec_h = self._compute_attention_score(
            q_head_h, k_head_h, v_head_h, k_head_r, segment_mat, attn_mask_h, **kwargs)

        # Post attention processing.
        attn_out_h = self._post_attention(attn_vec_h)

        output_h = tf.contrib.layers.layer_norm(
            attn_out_h + states_h, begin_norm_axis=-1,
            scope=self.variable_scope)

        if states_g is not None:
            heads_g = self.head_projection(states_g)
            q_head_g, _, _ = tf.split(heads_g, num_or_size_splits=3, axis=-1)
            shape = tf.shape(q_head_g)
            q_head_g = tf.reshape(
                shape[0], batch_size, self.num_heads, self.head_dim)
            if target_mapping is not None:
                q_head_g = tf.einsum(
                    'mbnd,mlb->lbnd', q_head_g, target_mapping)
            attn_vec_g = self._compute_attention_score(
                q_head_g, k_head_h, v_head_h, k_head_r,
                segment_mat, attn_mask_g, **kwargs)
            if target_mapping is not None:
                attn_vec_g = tf.einsum(
                    'lbnd,mlb->mbnd', attn_vec_g, target_mapping)
            attn_out_g = self._post_attention(attn_vec_g)
            output_g = tf.contrib.layers.layer_norm(
                attn_out_g + states_g, begin_norm_axis=-1,
                scope=self.variable_scope)
        else:
            output_g = None

        return output_h, output_g


class XLNetEncoder(EncoderBase):
    def __init__(self,
                 pretrained_model_name=None,
                 cache_dir=None,
                 hparams=None):
        EncoderBase.__init__(self, hparams)
        if pretrained_model_name:
            self.pretrained_model = bert_utils.\
                load_pretrained_model(pretrained_model_name, cache_dir)
        elif self._hparams.pretrained_model_name is not None:
            self.pretrained_model = bert_utils.\
                load_pretrained_model(self._hparams.pretrained_model_name,
                                      cache_dir)
        else:
            self.pretrained_model = None

        if self.pretrained_model:
            self.pretrained_model_hparams = bert_utils.\
                transform_xlnet_to_texar_config(self.pretrained_model)
            self._hparams = HParams(self.pretrained_model_hparams,
                                    self._hparams.todict())

        num_layers = self._hparams.num_layers

        with tf.variable_scope(self.variable_scope):

            if not self._hparams.untie_r:
                self.r_w_bias = tf.get_variable('r_w_bias',
                                                [self._hparams.num_heads,
                                                 self._hparams.head_dim],
                                                dtype=tf.float32)
                self.r_r_bias = tf.get_variable('r_r_bias',
                                                [self._hparams.num_heads,
                                                 self._hparams.head_dim],
                                                dtype=tf.float32)
                self.r_s_bias = tf.get_variable('r_s_bias',
                                                [self._hparams.num_heads,
                                                 self._hparams.head_dim],
                                                dtype=tf.float32)
            else:
                self.r_w_bias = None
                self.r_r_bias = None
                self.r_s_bias = None
            # Word embedding
            self.word_embedder = WordEmbedder(
                vocab_size=self._hparams.vocab_size,
                hparams={"dim": self._hparams.hidden_dim})

            # Position embedding
            self.pos_embed = RelativePositionalEncoding(hparams={
                "dim": self._hparams.hidden_dim,
                "max_seq_len": self._hparams.max_seq_len
            })

            self.attn_layers = []
            self.ff_layers = []
            rel_attn_hparams = dict_fetch(
                self._hparams, RelativeMutiheadAttention.default_hparams())

            ff_hparams = dict_fetch(
                self._hparams, PositionWiseFF.default_hparams())

            for _ in range(num_layers):
                self.attn_layers.append(RelativeMutiheadAttention(
                    self.r_r_bias, self.r_w_bias, self.r_s_bias,
                    hparams=rel_attn_hparams))
                self.ff_layers.append(PositionWiseFF(hparams=ff_hparams))

            self.dropout = tf.keras.layers.Dropout(rate=self._hparams.dropout)

    @staticmethod
    def default_hparams():

        return {
            "name": "xlnet_encoder",
            'pretrained_model_name': 'xlnet-large-cased',
            "untie_r": True,
            "num_layers": 24,
            "mem_len": 0,
            "reuse_len": 0,
            # initializer
            "initializer": None,
            # layer
            "num_heads": 16,
            "hidden_dim": 1024,
            "head_dim": 64,
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "use_segments": True,
            # ffn
            "ffn_inner_dim": 4096,
            "activation": 'relu',
            # embedding
            "vocab_size": 32000,
            "max_seq_len": 512,
            '@no_typecheck': ['pretrained_model_name']
        }

    @property
    def output_size(self):
        return self._hparams.hidden_dim

    @staticmethod
    def _cache_mem(curr_out, prev_mem, mem_len, reuse_len=None):
        r"""Cache hidden states into memory."""
        assert mem_len > 0

        if reuse_len is not None and reuse_len > 0:
            curr_out = curr_out[:reuse_len]

        if prev_mem is None:
            new_mem = curr_out[-mem_len:]
        else:
            new_mem = tf.concat([prev_mem, curr_out], 0)[-mem_len:]

        return tf.stop_gradient(new_mem)

    def _create_mask(self, qlen, mlen, dtype=tf.float32, same_length=False):
        r"""Create causal attention mask."""
        attn_mask = tf.ones([qlen, qlen], dtype=dtype)
        mask_u = tf.matrix_band_part(attn_mask, 0, -1)
        mask_dia = tf.matrix_band_part(attn_mask, 0, 0)
        attn_mask_pad = tf.zeros([qlen, mlen], dtype=dtype)
        ret = tf.concat([attn_mask_pad, mask_u - mask_dia], axis=1)
        if same_length:
            mask_l = tf.matrix_band_part(attn_mask, -1, 0)
            ret = tf.concat([ret[:, :qlen] + mask_l - mask_dia, ret[:, qlen:]],
                            axis=1)

        return ret

    def _build(self, inputs, segment_ids=None, input_mask=None, memory=None,
               permute_mask=None, bi_data=False, clamp_len=None, cache_len=0,
               same_length=False, attn_type='bi', **kwargs):
        r"""Encodes the inputs.
        """
        seq_len = tf.shape(inputs)[0]
        batch_size = tf.shape(inputs)[1]
        mem_len = tf.shape(memory[0])[0] if memory is not None else 0
        tot_len = seq_len + mem_len
        reuse_len = self._hparams.reuse_len
        is_training = kwargs["is_training"]

        # Attention mask
        # causal attention mask
        if attn_type == 'uni':
            attn_mask = self._create_mask(seq_len, mem_len, tf.float32,
                                          same_length)
            attn_mask = attn_mask[:, :, None, None]
        elif attn_type == 'bi':
            attn_mask = None
        else:
            raise ValueError('Unsupported attention type: {}'.format(attn_type))

        # data mask: input mask & perm mask
        if input_mask is not None and permute_mask is not None:
            data_mask = input_mask[None] + permute_mask
        elif input_mask is not None and permute_mask is None:
            data_mask = input_mask[None]
        elif input_mask is None and permute_mask is not None:
            data_mask = permute_mask
        else:
            data_mask = None

        if data_mask is not None:
            # all mems can be attended to
            mems_mask = tf.zeros([tf.shape(data_mask)[0], mem_len, batch_size],
                                 dtype=tf.float32)
            data_mask = tf.concat([mems_mask, data_mask], 1)
            if attn_mask is None:
                attn_mask = data_mask[:, :, :, None]
            else:
                attn_mask += data_mask[:, :, :, None]

        if attn_mask is not None:
            attn_mask = tf.cast(attn_mask > 0, dtype=tf.float32)

        if attn_mask is not None:
            non_tgt_mask = -tf.eye(seq_len, dtype=tf.float32)
            non_tgt_mask = tf.concat([tf.zeros([seq_len, mem_len],
                                               dtype=tf.float32),
                                      non_tgt_mask], axis=-1)
            non_tgt_mask = tf.cast(
                (attn_mask + non_tgt_mask[:, :, None, None]) > 0,
                dtype=tf.float32)
        else:
            non_tgt_mask = None

        # word embedding
        word_embed = self.word_embedder(inputs)

        # Segment embedding
        if segment_ids is not None:
            mem_pad = tf.zeros([mem_len, batch_size], dtype=tf.int32)
            cat_ids = tf.concat([mem_pad, segment_ids], 0)
            segment_matrix = tf.cast(
                tf.logical_not(tf.equal(segment_ids[:, None], cat_ids[None, :])),
                tf.int32)
            segment_matrix = tf.one_hot(segment_matrix, 2, dtype=tf.float32)
        else:
            segment_matrix = None

        # Position embedding
        pos_embed = self.pos_embed(
            batch_size, seq_len, tot_len, clamp_len, attn_type, bi_data)
        pos_embed = self.dropout(pos_embed, training=is_training)

        states_h = self.dropout(word_embed, training=is_training)

        new_memory = []
        for i in range(self._hparams.num_layers):
            cur_memory = memory[i] if memory is not None else None
            if cache_len > 0:
                new_memory.append(self._cache_mem(states_h, cur_memory, cache_len, reuse_len))
            states_h, _ = self.attn_layers[i](
                states_h=states_h, states_g=None, pos_embed=pos_embed,
                segment_mat=segment_matrix, attn_mask_h=non_tgt_mask,
                attn_mask_g=attn_mask, target_mapping=None, memory=cur_memory,
                is_training=is_training)
            states_h = self.ff_layers[i](states_h)

        if not self._built:
            self._add_internal_trainable_variables()
            self._built = True

            if self.pretrained_model:
                model_utils.init_from_checkpoint(self.pretrained_model,
                                                 self.variable_scope.name)

        output = self.dropout(states_h)
        return output

