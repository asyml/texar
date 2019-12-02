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

from texar.tf.core import layers
from texar.tf.module_base import ModuleBase
from texar.tf.utils.mode import is_train_mode


__all__ = [
    'PositionWiseFF',
    'PositionalEmbedding',
    'RelativePositionalEncoding',
    'RelativeMutiheadAttention'
]


class PositionWiseFF(ModuleBase):
    r"""Position Wise feed forward."""
    def __init__(self, hparams=None):
        ModuleBase.__init__(self, hparams)

        hidden_dim = self._hparams.hidden_dim
        ffn_inner_dim = self._hparams.ffn_inner_dim
        dropout = self._hparams.dropout
        activation = self._hparams.activation
        if activation == 'gelu':
            activation = layers.gelu

        with tf.variable_scope(self.variable_scope):
            tf.get_variable_scope().set_initializer(
                layers.get_initializer(self._hparams.initializer))
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
                    "units": hidden_dim
                }
            }
            self.linear2 = layers.get_layer(hparams=l2_hparams)

    @staticmethod
    def default_hparams():
        r"""Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                "hidden_dim": 768,
                "ffn_inner_dim": 3072,
                "dropout": 0.1,
                "activation": 'gelu'
            }

        Here

        `"hidden_dim"`: int
            Dimension of the layer fed as input to feed forward network

        `"ffn_inner_dim"`: int
            Inner dimension of the feed forward layer

        `"dropout"`: float
            Dropout rate for layers

        `"activation"`: str or callable
            Activation function applied to the output of the PositionWise FF.
            See :func:`~texar.tf.core.get_activation_fn` for more details.
        """
        return {
            "name": "ff",
            "initializer": None,
            "hidden_dim": 768,
            "ffn_inner_dim": 3072,
            "dropout": 0.1,
            "activation": 'gelu',
        }

    def _build(self, input, mode=None):
        r"""Compute feed forward for the input.

        Args:
            input: Input tensor of size `(max_time, batch_size, hidden_dim)`
            mode (optional): A tensor taking value in
                :tf_main:`tf.estimator.ModeKeys <estimator/ModeKeys>`, including
                `TRAIN`, `EVAL`, and `PREDICT`. If `None`, dropout is
                controlled by :func:`texar.tf.global_mode`.

        :returns: A tensor output of the position wise feed forward network
        """
        is_training = is_train_mode(mode)
        output = self.linear1(input)
        output = self.dropout(output, training=is_training)
        output = self.linear2(output)
        output = self.dropout(output, training=is_training)

        # residual + layer norm
        output = tf.contrib.layers.layer_norm(
            input + output, begin_norm_axis=-1, scope=self.variable_scope,
            reuse=tf.AUTO_REUSE)

        return output


class PositionalEmbedding(ModuleBase):
    r"""Sinosoidal Positional Embedding.
    """

    # TODO(avinash) : See if this can be merged with Sinosoidal Position
    # Embedder
    def __init__(self, embed_dim):
        ModuleBase.__init__(self)
        freq_seq = tf.range(0.0, embed_dim, 2.0)
        self.inv_freq = 1 / (10000 ** (freq_seq / embed_dim))

    def _build(self, pos_seq):
        r"""Compute sinosoidal positional embeddings.

        Args:
            pos_seq: A 1D tensor of position sequences

        :returns: A 2D tensor of sinosoidal embeddings for the sequence.
        """
        pos_seq = tf.dtypes.cast(pos_seq, dtype=self.inv_freq.dtype)
        sinusoid_inp = tf.einsum('i,d->id', pos_seq, self.inv_freq)
        pos_emb = tf.concat([tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)], -1)
        return pos_emb


class RelativePositionalEncoding(ModuleBase):
    r"""Relative positional encodings."""
    def __init__(self, hparams=None):
        ModuleBase.__init__(self, hparams)
        self.sinusoid_embed = PositionalEmbedding(self._hparams.dim)

    @staticmethod
    def default_hparams():
        r"""Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                "dim": 768,
                "max_seq_len": 512
            }

        Here

        `"dim"`: int
            Dimension size of the positional embedding

        `"max_seq_len"`: int
            Maximum size of the sequence length
        """
        return {
            "name": "relative_positional_encoder",
            "dim": 768,
            "max_seq_len": 512
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

    def _build(self, batch_size, max_time, total_len, clamp_len=None,
               attn_type='bi', bi_data=True):
        r"""Compute relative positional encoding.

        Args
            batch_size: int
                Batch size of the input

            max_time: int
                Sequence length of the input

            total_len: int
                Sequence length + Memory length

            clamp_len (optional): int
                Clamp all relative distances larger than clamp_len.
                None means no clamping.

            attn_type (optional): str
                Attention type. Supported values are `"uni"` and `"bi"`.

            bi_data (optional): bool
                Whether to use bidirectional data input pipeline. Usually set to
                True during pretraining and False during finetuning.

        :returns: A tensor of shape `[total_len + max_time, batch_size, dim]`
            (if attn_type == `"bi"`) or of shape `[total_len, batch_size, dim]`
            (if attn_type == `"uni"`) representing relative positional encoding
            of the sequence.
        """
        if attn_type == 'bi':
            start, end = total_len, -max_time
        elif attn_type == 'uni':
            start, end = total_len, -1
        else:
            raise ValueError("Unknown `attn_type` {}".format(attn_type))

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
    r"""Compute relative multi-head attention for XLNet encoder.

    This module computes relative multi-head attention as explained in
    `Transformer-XL, (Zihang et. al)` and in `XLNet (Zhiling et. al)`.

    Args:
        r_r_bias: A tensor of shape `(num_heads, head_dim)`.
            The bias value added to query head while computing position based
            attention score.

        r_w_bias: A tensor of shape `(num_heads, head_dim)`.
            The bias value added to query head while computing content based
            attention score.

        r_s_bias (optional): A tensor of shape `(num_heads, head_dim)`.
            The bias value added to query head while computing segment based
            attention score.

        segment_embed (optional): A tensor of shape `(2, num_heads, head_dim)`
            if use_segments is True. Otherwise, this is set to None.

        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparameter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter sturcture
            and default values.
    """
    def __init__(self, r_r_bias, r_w_bias, r_s_bias=None, segment_embed=None,
                 hparams=None):
        ModuleBase.__init__(self, hparams=hparams)

        self.num_heads = self._hparams.num_heads
        self.head_dim = self._hparams.head_dim
        hidden_dim = self._hparams.hidden_dim

        with tf.variable_scope(self.variable_scope):
            if self._hparams.initializer:
                tf.get_variable_scope().set_initializer(
                    layers.get_initializer(self._hparams.initializer))

            # Official implementation creates these head variables.
            # If we create dense layers instead, there would be dimension
            # mismatch while loading the tensors
            # TODO(avinash) : Can we reshape tensors while loading the ckpt?
            self.q_head = tf.get_variable(
                'q/kernel', [hidden_dim, self.num_heads, self.head_dim])

            self.k_head = tf.get_variable(
                'k/kernel', [hidden_dim, self.num_heads, self.head_dim])

            self.v_head = tf.get_variable(
                'v/kernel', [hidden_dim, self.num_heads, self.head_dim])

            self.k_head_r = tf.get_variable(
                'r/kernel', [hidden_dim, self.num_heads, self.head_dim])

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

            self.output_projection = tf.get_variable(
                'o/kernel', [hidden_dim, self.num_heads, self.head_dim])

            self.r_r_bias = r_r_bias
            self.r_w_bias = r_w_bias

            if self._hparams.use_segments:
                self.segment_embed = segment_embed
                self.r_s_bias = r_s_bias

            self.scale = 1 / (self.head_dim ** 0.5)

    @staticmethod
    def default_hparams():
        r"""Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                "name": "rel_attn",
                "initializer": None,
                "num_heads": 12,
                "hidden_dim": 768,
                "head_dim": 64,
                "dropout": 0.1,
                "attention_dropout": 0.1,
                "use_segments": True
            }



        Here:

        The default parameters are values for cased XLNet-Base model.

        "initializer": dict, optional
            Hyperparameters of the default initializer that initializes
            variables created in this module.
            See :func:`~texar.tf.core.get_initializer` for details.

        "num_heads": int
            Number of heads in the attention

        "hidden_dim": int
            Hidden dimension of the embeddings

        "head_dim": int
            Size of the vectors after head projection.

        "dropout": float
            Dropout rate for layers

        "attention_dropout": float
            Dropout rate for attention layers

        "use_segments": bool
            Boolean to indicate if the input has segments

        "name": str
            Name of the module.
        """
        return {
            "name": "rel_attn",
            "initializer": None,
            "num_heads": 12,
            "hidden_dim": 768,
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
        # attn_ac: (max_time, tot_len, batch_size, n_head)
        attn_ac = tf.einsum('ibnd,jbnd->ijbn', q_head_rw, k_head_h)

        # Position based attention score.
        q_head_rr = q_head + self.r_r_bias
        # attn_bd: (max_time, tot_len, batch_size, n_head)
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
        # attn_score: (max_time, tot_len, batch_size, n_head)
        attn_score = (attn_ac + attn_bd + attn_ef) * self.scale
        if attn_mask is not None:
            # attn_score = attn_score * (1 - attn_mask) - 1e30 * attn_mask
            attn_score = attn_score - 1e30 * attn_mask

        # attention probability
        attn_prob = tf.nn.softmax(attn_score, 1)
        attn_prob = self.dropout_attn(attn_prob, training=is_training)

        # attention output
        attn_vec = tf.einsum('ijbn,jbnd->ibnd', attn_prob, v_head_h)

        return attn_vec

    def _post_attention(self, attn_vec, mode=None):
        is_training = is_train_mode(mode)
        attn_out = tf.einsum('ibnd,hnd->ibh', attn_vec, self.output_projection)
        attn_out = self.dropout(attn_out, training=is_training)
        return attn_out

    def _build(self, states_h, pos_embed, states_g=None, segment_mat=None,
               attn_mask_h=None, attn_mask_g=None, target_mapping=None,
               memory=None, mode=None):
        r"""Compute relative multi-head attention with relative positional
        encoding.

        Args:
            states_h: A content representation tensor of shape
                `[max_time, batch_size, hidden_dim]`

            pos_embed: Position embedding tensor of shape
                `[max_time, batch_size, hidden_dim]`.

            states_g (optional): A query representation tensor of shape
                `[max_time, batch_size, hidden_dim]`. This tensor is set during
                decoding.

            segment_mat (optional): A tensor of size
                `[max_time, tot_len, batch_size]` indicating if tokens are in the
                 same seqment. A value at `(i, j, k)` of `1` indicates tokens at
                  `i` and `j` are not in the same sequence in batch k.

            attn_mask_h (optional): A tensor of shape
                `[max_time, max_time, batch_size, 1]` Attention mask used while
                computing attention score for `states_h`

            attn_mask_g (optional): A tensor of shape
                `[max_time, max_time, batch_size, 1]` Attention mask used while
                computing attention score for `states_g`

            target_mapping (optional): The target token mapping. Float tensor of
                shape `[num_targets, max_time, batch_size]`.
                A value of 1 for ``target_mapping[i, j, k]`` indicates that
                the `i`-th target token (in order of permutation) in batch `k`
                is the token at position `j`.
                Each row ``target_mapping[i, :, k]`` can have no more than one
                value of 1.

            memory (optional): Memory from previous batches. A list of length
                `num_layers`, each a tensor of shape
                `[mem_len, batch_size, hidden_dim]`.

            mode (optional): A tensor taking value in
                :tf_main:`tf.estimator.ModeKeys <estimator/ModeKeys>`, including
                `TRAIN`, `EVAL`, and `PREDICT`. If `None`, dropout is
                controlled by :func:`texar.tf.global_mode`.

        :returns: Returns output states for `states_h` and `states_g`
            (`states_g` is not None)
        """
        batch_size = tf.shape(states_h)[1]

        if memory is not None and memory.shape.ndims > 1:
            concat_input = tf.concat([memory, states_h], axis=0)
        else:
            concat_input = states_h

        # Content heads.
        q_head_h = tf.einsum('ibh,hnd->ibnd', states_h, self.q_head)
        k_head_h = tf.einsum('ibh,hnd->ibnd', concat_input, self.k_head)
        v_head_h = tf.einsum('ibh,hnd->ibnd', concat_input, self.v_head)

        # Positional heads.
        k_head_r = tf.einsum('ibh,hnd->ibnd', pos_embed, self.k_head_r)

        # Core attention ops.
        attn_vec_h = self._compute_attention_score(
            q_head_h, k_head_h, v_head_h, k_head_r, segment_mat, attn_mask_h,
            mode)

        # Post attention processing.
        attn_out_h = self._post_attention(attn_vec_h, mode=mode)

        output_h = tf.contrib.layers.layer_norm(
            attn_out_h + states_h, begin_norm_axis=-1,
            scope=self.variable_scope, reuse=tf.AUTO_REUSE)

        if states_g is not None:
            q_head_g = tf.einsum('ibh,hnd->ibnd', states_g, self.q_head)
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
            attn_out_g = self._post_attention(attn_vec_g, mode=mode)
            output_g = tf.contrib.layers.layer_norm(
                attn_out_g + states_g, begin_norm_axis=-1,
                scope=self.variable_scope, reuse=tf.AUTO_REUSE)
        else:
            output_g = None

        return output_h, output_g
