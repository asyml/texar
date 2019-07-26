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
Model Utils of XLNet Modules.
Adapted from
https://github.com/zihangdai/xlnet/blob/master/modeling.py
"""

import tensorflow as tf

from texar.core import layers
from texar.utils.mode import is_train_mode
from texar.module_base import ModuleBase


__all__ = [
    'PositionWiseFF',
    'PositionalEmbedding',
    'RelativePositionalEncoding',
    'RelativeMutiheadAttention'
]


class PositionWiseFF(ModuleBase):
    r"""Position Wise feed forward."""
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
        r"""Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                "hidden_dim": 1024,
                "ffn_inner_dim": 4096,
                "dropout": 0.1,
                "activation": 'gelu'
            }

        Here

        "hidden_dim": int
            Dimension of the layer fed as input to feed forward network

        "ffn_inner_dim": int
            Inner dimension of the feed forward layer

        "dropout": float
            Dropout rate for layers

        "activation": str or callable
            Activation function applied to the output of the PositionWise FF.
            See :func:`~texar.core.get_activation_fn` for more details.
        """
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
            input + output, begin_norm_axis=-1, scope=self.variable_scope,
            reuse=tf.AUTO_REUSE)
        return output


class PositionalEmbedding(ModuleBase):
    """Sinosoidal Positional Embedding."""
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
    def __init__(self, r_r_bias=None, r_w_bias=None, r_s_bias=None,
                 hparams=None):
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
                                 segment_mat, attn_mask=None, mode=None):
        is_training = is_train_mode(mode)

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
        attn_prob = self.dropout(attn_prob,
                                 training=is_training)

        # attention output
        attn_vec = tf.einsum('ijbn,jbnd->ibnd', attn_prob, v_head_h)

        return attn_vec

    def _post_attention(self, attn_vec):
        shape = tf.shape(attn_vec)
        attn_vec = tf.reshape(
            tensor=attn_vec,
            shape=[shape[0], shape[1], self.num_heads*self.head_dim])
        attn_out = self.output_projection(attn_vec)
        attn_out = self.dropout(attn_out)
        return attn_out

    def _build(self, states_h, pos_embed, states_g=None, segment_mat=None,
               attn_mask_h=None, attn_mask_g=None, target_mapping=None,
               memory=None, mode=None):
        shape = tf.shape(states_h)
        seq_len, batch_size = shape[0], shape[1]
        pos_len = tf.shape(pos_embed)[0]

        if memory is not None and memory.shape.ndims > 1:
            concat_input = tf.concat([memory, states_h], axis=0)
        else:
            concat_input = states_h

        # Content heads.
        heads = self.head_projection(concat_input)
        q_head_h, k_head_h, v_head_h = tf.split(
            heads, num_or_size_splits=3, axis=-1)
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
            q_head_h, k_head_h, v_head_h, k_head_r, segment_mat, attn_mask_h,
            mode)

        # Post attention processing.
        attn_out_h = self._post_attention(attn_vec_h)

        output_h = tf.contrib.layers.layer_norm(
            attn_out_h + states_h, begin_norm_axis=-1,
            scope=self.variable_scope, reuse=tf.AUTO_REUSE)

        if states_g is not None:
            heads_g = self.head_projection(states_g)
            q_head_g, _, _ = tf.split(heads_g, num_or_size_splits=3, axis=-1)
            shape = tf.shape(q_head_g)
            q_head_g = tf.reshape(
                q_head_g,
                shape=(shape[0], batch_size, self.num_heads, self.head_dim))
            if target_mapping is not None:
                q_head_g = tf.einsum(
                    'mbnd,mlb->lbnd', q_head_g, target_mapping)
            attn_vec_g = self._compute_attention_score(
                q_head_g, k_head_h, v_head_h, k_head_r,
                segment_mat, attn_mask_g, mode)
            if target_mapping is not None:
                attn_vec_g = tf.einsum(
                    'lbnd,mlb->mbnd', attn_vec_g, target_mapping)
            attn_out_g = self._post_attention(attn_vec_g)
            output_g = tf.contrib.layers.layer_norm(
                attn_out_g + states_g, begin_norm_axis=-1,
                scope=self.variable_scope, reuse=tf.AUTO_REUSE)
        else:
            output_g = None

        return output_h, output_g
