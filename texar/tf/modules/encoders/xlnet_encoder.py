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

import tensorflow as tf

from texar.tf.utils.mode import is_train_mode

from texar.tf.core.layers import get_initializer, get_layer
from texar.tf.modules.embedders.embedders import WordEmbedder
from texar.tf.modules.encoders.encoder_base import EncoderBase
from texar.tf.modules.pretrained.xlnet import PretrainedXLNetMixin
from texar.tf.modules.pretrained.xlnet_utils import \
    (PositionWiseFF, RelativePositionalEncoding, RelativeMutiheadAttention)
from texar.tf.utils.utils import dict_fetch

__all__ = [
    "XLNetEncoder"
]


class XLNetEncoder(EncoderBase, PretrainedXLNetMixin):
    r"""Raw XLNet module for encoding sequences. Please see
    :class:`~texar.tf.modules.PretrainedXLNetMixin` for a brief description
    of XLNet.

    Args:
        pretrained_model_name (optional): a `str`, the name
            of pre-trained model (e.g., ``xlnet-based-cased``). Please refer to
            :class:`~texar.tf.modules.PretrainedXLNetMixin` for
            all supported models.
            If `None`, the model name in :attr:`hparams` is used.
        cache_dir (optional): the path to a folder in which the
            pre-trained models will be cached. If `None` (default),
            a default directory (``texar_data`` folder under user's home
            directory) will be used.
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparameter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter structure
            and default values.

    .. document private functions
    .. automethod:: _build
    """

    def __init__(self,
                 pretrained_model_name=None,
                 cache_dir=None,
                 hparams=None):
        super(XLNetEncoder, self).__init__(hparams=hparams)

        self.load_pretrained_config(pretrained_model_name, cache_dir)

        num_layers = self._hparams.num_layers
        use_segments = self._hparams.use_segments
        untie_r = self._hparams.untie_r

        with tf.variable_scope(self.variable_scope):

            if untie_r:
                self.r_w_bias = tf.get_variable('r_w_bias',
                                                [num_layers,
                                                 self._hparams.num_heads,
                                                 self._hparams.head_dim],
                                                dtype=tf.float32)
                self.r_r_bias = tf.get_variable('r_r_bias',
                                                [num_layers,
                                                 self._hparams.num_heads,
                                                 self._hparams.head_dim],
                                                dtype=tf.float32)
            else:
                self.r_w_bias = tf.get_variable('r_w_bias',
                                                [self._hparams.num_heads,
                                                 self._hparams.head_dim],
                                                dtype=tf.float32)
                self.r_r_bias = tf.get_variable('r_r_bias',
                                                [self._hparams.num_heads,
                                                 self._hparams.head_dim],
                                                dtype=tf.float32)

            if use_segments:
                self.segment_embed = tf.get_variable('seg_embed',
                                                     [num_layers, 2,
                                                      self._hparams.num_heads,
                                                      self._hparams.head_dim],
                                                     dtype=tf.float32)
                self.r_s_bias = (tf.get_variable('r_s_bias',
                                                 [num_layers,
                                                  self._hparams.num_heads,
                                                  self._hparams.head_dim],
                                                 dtype=tf.float32) if untie_r
                                 else tf.get_variable('r_s_bias',
                                                      [self._hparams.num_heads,
                                                       self._hparams.head_dim],
                                                      dtype=tf.float32))
            else:
                self.segment_embed = None
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
            rel_attn_hparams["name"] = "rel_attn"

            ff_hparams = dict_fetch(
                self._hparams, PositionWiseFF.default_hparams())
            ff_hparams["name"] = "ff"

            for i in range(num_layers):
                with tf.variable_scope("layer_{}".format(i)):
                    if self._hparams.untie_r:
                        if use_segments:
                            self.attn_layers.append(RelativeMutiheadAttention(
                                self.r_r_bias[i], self.r_w_bias[i],
                                self.r_s_bias[i],
                                self.segment_embed[i],
                                hparams=rel_attn_hparams))
                        else:
                            self.attn_layers.append(RelativeMutiheadAttention(
                                self.r_r_bias[i], self.r_w_bias[i],
                                hparams=rel_attn_hparams))
                    else:
                        if use_segments:
                            self.attn_layers.append(RelativeMutiheadAttention(
                                self.r_r_bias, self.r_w_bias,
                                self.r_s_bias,
                                self.segment_embed[i],
                                hparams=rel_attn_hparams))
                        else:
                            self.attn_layers.append(RelativeMutiheadAttention(
                                self.r_r_bias, self.r_w_bias,
                                hparams=rel_attn_hparams))
                    self.ff_layers.append(PositionWiseFF(hparams=ff_hparams))

            dropout_hparams = {
                "type": "Dropout",
                "kwargs": {
                    "rate": self._hparams.dropout
                }
            }
            self.dropout = get_layer(hparams=dropout_hparams)

            self.mask_embed = tf.get_variable(
                'mask_emb', [1, 1, self.hparams.hidden_dim], dtype=tf.float32)

    def reset_parameters(self):
        with tf.variable_scope(self.variable_scope):
            if self._hparams.initializer:
                tf.get_variable_scope().set_initializer(
                    get_initializer(self._hparams.initializer))

    @staticmethod
    def default_hparams():
        r"""Returns a dictionary of hyperparameters with default values.

        * The encoder arch is determined by the constructor argument
        :attr:`pretrained_model_name` if it's specified. In this case,
        hparams are ignored.
        * Otherwise, the encoder arch is determined by
        `hparams['pretrained_model_name']` if it's specified. All other
        configs in hparams are ignored.
        * If the above two are `None`, the encoder arch is defined by
        the configs in hparams and weights are randomly initialized.

        .. code-block:: python

            {
                "name": "xlnet_encoder",
                "pretrained_model_name": "xlnet-base-cased",
                "untie_r": True,
                "num_layers": 12,
                "mem_len": 0,
                "reuse_len": 0,
                "initializer": None,
                "num_heads": 12,
                "hidden_dim": 768,
                "head_dim": 64,
                "dropout": 0.1,
                "attention_dropout": 0.1,
                "use_segments": True,
                "ffn_inner_dim": 3072,
                "activation": 'gelu',
                "vocab_size": 32000,
                "max_seq_len": 512,
            }



        Here:

        The default parameters are values for cased XLNet-Base model.


        "pretrained_model_name": str or None
             The name of the pre-trained bert model. If None, the model
             will be randomly initialized.

        "untie_r": bool
            Boolean value to indicate if biases should be untied for all the
            layers

        "num_layers": int
            Number of layers in the network

        "mem_len": int
            Length of the memory to be used during attention score calculation.

        "reuse_len": int
            Length of the memory that can be re-used

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

        "ffn_inner_dim": int
            Dimension of PositionWise FF network's hidden layer

        "activation": str or callable
            Activation function applied to the output of the PositionWise FF.
            See :func:`~texar.tf.core.get_activation_fn` for more details.

        "vocab_size": int
            The vocabulary size of `inputs` in `XLNet`.

        "max_seq_len": int
            Maximum len of the sequence allowed in one segment

        "name": str
            Name of the module.
        """

        return {
            "name": "xlnet_encoder",
            'pretrained_model_name': 'xlnet-base-cased',
            "untie_r": True,
            "num_layers": 12,
            "mem_len": 0,
            "reuse_len": 0,
            # initializer
            "initializer": None,
            # layer
            "num_heads": 12,
            "hidden_dim": 768,
            "head_dim": 64,
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "use_segments": True,
            # ffn
            "ffn_inner_dim": 3072,
            "activation": 'gelu',
            # embedding
            "vocab_size": 32000,
            "max_seq_len": 512,
            '@no_typecheck': ['pretrained_model_name']
        }

    def param_groups(self, lr=None, lr_layer_scale=1.0,
                     decay_base_params=False):
        r"""Create parameter groups for optimizers. When
        :attr:`lr_layer_decay_rate` is not 1.0, parameters from each layer form
        separate groups with different base learning rates.

        This method should be called before applying gradients to the variables
        through the optimizer. Particularly, after calling the optimizer's
        `compute_gradients` method, the user can call this method to get
        variable-specific learning rates for the network. The gradients for each
        variables can then be scaled accordingly. These scaled gradients are
        finally applied by calling optimizer's `apply_gradients` method.

        Example:

            .. code-block:: python

            grads_and_vars = optimizer.compute_gradients(loss)

            vars_to_grads = {key: value for key, value in grads_and_vars}

            vars_to_learning_rates = xlnet_encoder.param_groups(
                                        lr=1, ly_layer_scale=0.75)

            for key in vars_to_grads.keys():
                vars_to_grads[key] *= vars_to_learning_rates[key]

            train_op = optimizer.apply_gradients(zip(*vars_to_grads.items()))


        Args:
            lr (float): The learning rate. Can be omitted if
                :attr:`lr_layer_decay_rate` is 1.0.
            lr_layer_scale (float): Per-layer LR scaling rate. The `i`-th layer
                will be scaled by `lr_layer_scale ^ (num_layers - i - 1)`.
            decay_base_params (bool): If `True`, treat non-layer parameters
                (e.g. embeddings) as if they're in layer 0. If `False`, these
                parameters are not scaled.

        Returns: A dict mapping tensorflow variables to their learning rates.
        """
        vars_to_learning_rates = {}
        if lr_layer_scale != 1.0:
            if lr is None:
                raise ValueError(
                    "lr must be specified when lr_layer_decay_rate is not 1.0")

            num_layers = self._hparams.num_layers
            scope = self.variable_scope.name
            base_var_names = ['r_w_bias', 'r_r_bias', 'word_embedder']

            if self._hparams.use_segments:
                base_var_names.extend(['r_s_bias', 'seg_embed'])

            for var in base_var_names:
                tf_variable = tf.trainable_variables(scope=scope + "/" + var)[0]
                vars_to_learning_rates[tf_variable] = \
                    lr * (lr_layer_scale ** num_layers if decay_base_params
                          else 1.0)

            for idx in range(num_layers):
                decay_rate = lr_layer_scale ** (num_layers - idx - 1)
                layer_variables = tf.trainable_variables(
                    scope=scope + "/" + "layer_{}".format(idx))
                for variable in layer_variables:
                    vars_to_learning_rates[variable] = lr * decay_rate
        else:
            for variable in self.trainable_variables:
                vars_to_learning_rates[variable] = lr

        return vars_to_learning_rates

    @property
    def output_size(self):
        r"""The last dimension of the encoder output.

        Note: The :meth:`_build` returns two tensors of shapes
        `[batch_size, max_time, hidden_dim]` and
        `[batch_size, cache_len, hidden_dim]`. `output_size` here equals
        `hidden_dim`
        """
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

    def _build(self, token_ids, segment_ids=None, input_mask=None,
               memory=None, permute_mask=None, target_mapping=None,
               bi_data=False, clamp_len=None, cache_len=0, same_length=False,
               attn_type='bi', two_stream=False, mode=None):
        r"""Compute XLNet representations for the input.

        Args:
            token_ids: Shape `[batch_size, max_time]`.
            segment_ids: Shape `[batch_size, max_time]`.
            input_mask: Float tensor of shape `[batch_size, max_time]`. Note that
                positions with value 1 are masked out.
            memory: Memory from previous batches. A list of length `num_layers`,
                each tensor of shape `[batch_size, mem_len, hidden_dim]`.
            permute_mask: The permutation mask. Float tensor of shape
                `[batch_size, max_time, max_time]`.
                A value of 0 for ``permute_mask[i, j, k]`` indicates that
                position `i` attends to position `j` in batch `k`.
            target_mapping: The target token mapping. Float tensor of shape
                `[batch_size, num_targets, max_time]`.
                A value of 1 for ``target_mapping[i, j, k]`` indicates that
                the `i`-th target token (in order of permutation) in batch `k`
                is the token at position `j`.
                Each row ``target_mapping[i, :, k]`` can have no more than one
                value of 1.
            bi_data (bool): Whether to use bidirectional data input pipeline.
            clamp_len (int): Clamp all relative distances larger than
                :attr:`clamp_len`. A value of -1 means no clamping.
            cache_len (int): Length of memory (number of tokens) to cache.
            same_length (bool): Whether to use the same attention length for
                each token.
            attn_type (str): Attention type. Supported values are `"uni"`
                and `"bi"`.
            two_stream (bool): Whether to use two-stream attention. Only set to
                `True` when pre-training or generating text. Defaults to
                `False`.

        Returns: A tuple of `(output, new_memory)`:

            - **output**: The final layer output representations. Shape
              `[batch_size, max_time, hidden_dim]`.
            - **new_memory**: The memory of the current batch.
              If `cache_len` is 0, then `new_memory` is `None`. Otherwise, it is
              a list of length `num_layers`, each tensor of shape
              `[batch_size, cache_len, hidden_dim]`.
              This can be used as the :attr:`memory` argument in the next batch.
        """
        return self._execute(self.word_embedder(token_ids),
                             segment_ids=segment_ids, input_mask=input_mask,
                             memory=memory, permute_mask=permute_mask,
                             target_mapping=target_mapping, bi_data=bi_data,
                             clamp_len=clamp_len, cache_len=cache_len,
                             same_length=same_length, attn_type=attn_type,
                             two_stream=two_stream, mode=mode)

    def _execute(self, word_embed, segment_ids=None,  # noqa: C901
                 input_mask=None, memory=None, permute_mask=None,
                 target_mapping=None, bi_data=False, clamp_len=None,
                 cache_len=0, same_length=False, attn_type='bi',
                 two_stream=False, mode=None):
        r"""Compute XLNet representations for the input. This layer exists
        because :class:`XLNetDecoder` compute embeddings in the decoder helper.
        `word_embed` has shape `[batch_size, max_time, word_embed_dim]`.
        Please refer to :meth:`_build` for the detailed information of other
        arguments.
        """
        # word_embed: [max_time, batch_size, word_embed_dim]
        word_embed = tf.transpose(word_embed, perm=[1, 0, 2])
        # segment_ids: [max_time, batch_size]
        if segment_ids is not None:
            segment_ids = tf.transpose(segment_ids, perm=[1, 0])
        # input_mask: [max_time, batch_size]
        if input_mask is not None:
            input_mask = tf.transpose(input_mask, perm=[1, 0])
        # memory: A list of length num_layers
        # each tensor of shape [mem_len, batch_size, hidden_dim]
        if memory is not None:
            memory = [tf.transpose(m, perm=[1, 0, 2]) for m in memory]
        # permute_mask: [max_time, max_time, batch_size]
        if permute_mask is not None:
            permute_mask = tf.transpose(permute_mask, perm=[1, 2, 0])
        # target_mapping: [num_targets, max_time, batch_size]
        if target_mapping is not None:
            target_mapping = tf.transpose(target_mapping, perm=[1, 2, 0])

        max_time = tf.shape(word_embed)[0]
        batch_size = tf.shape(word_embed)[1]
        mem_len = tf.shape(memory[0])[0] if memory is not None else 0
        tot_len = max_time + mem_len
        reuse_len = self._hparams.reuse_len
        is_training = is_train_mode(mode)

        # Attention mask
        # causal attention mask
        if attn_type == 'uni':
            attn_mask = self._create_mask(max_time, mem_len, tf.float32,
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
            non_tgt_mask = -tf.eye(max_time, dtype=tf.float32)
            non_tgt_mask = tf.concat([tf.zeros([max_time, mem_len],
                                               dtype=tf.float32),
                                      non_tgt_mask], axis=-1)
            non_tgt_mask = tf.cast(
                (attn_mask + non_tgt_mask[:, :, None, None]) > 0,
                dtype=tf.float32)
        else:
            non_tgt_mask = None

        # Segment embedding
        if segment_ids is not None:
            mem_pad = tf.zeros([mem_len, batch_size], dtype=tf.int32)
            cat_ids = tf.concat([mem_pad, segment_ids], 0)
            segment_matrix = tf.cast(
                tf.logical_not(
                    tf.equal(segment_ids[:, None], cat_ids[None, :])),
                tf.int32)
            segment_matrix = tf.one_hot(segment_matrix, 2, dtype=tf.float32)
        else:
            segment_matrix = None

        # Position embedding
        pos_embed = self.pos_embed(
            batch_size, max_time, tot_len, clamp_len, attn_type, bi_data)
        pos_embed = self.dropout(pos_embed,
                                 training=is_training)

        states_h = self.dropout(word_embed,
                                training=is_training)

        if two_stream:
            if target_mapping is not None:
                word_embed_q = tf.tile(
                    self.mask_embed, [tf.shape(target_mapping)[0],
                                      batch_size, 1])
            else:
                word_embed_q = word_embed
            states_g = self.dropout(word_embed_q)
        else:
            states_g = None

        new_memory = []
        num_layers = self._hparams.num_layers
        for i in range(num_layers):
            cur_memory = memory[i] if memory is not None else None
            if cache_len > 0:
                new_memory.append(
                    self._cache_mem(states_h, cur_memory, cache_len, reuse_len))
            states_h, states_g = self.attn_layers[i](
                states_h=states_h, pos_embed=pos_embed, states_g=states_g,
                segment_mat=segment_matrix, attn_mask_h=non_tgt_mask,
                attn_mask_g=attn_mask, target_mapping=None, memory=cur_memory,
                mode=mode)
            ff_layer = self.ff_layers[i]
            states_h = ff_layer(states_h, mode=mode)

            if states_g is not None:
                states_g = ff_layer(states_g, mode=mode)

        output = self.dropout(states_h if states_g is None else states_g,
                              training=is_training)

        # Now output: [max_time, batch_size, hidden_dim]
        # new_memory: None or A list of length num_layers,
        # each tensor of shape [cache_len, batch_size, hidden_dim]
        output = tf.transpose(output, perm=[1, 0, 2])
        if new_memory is not None:
            new_memory = [tf.transpose(m, perm=[1, 0, 2]) for m in new_memory]

        if not self._built:
            self._add_internal_trainable_variables()
            self._built = True

            if self.pretrained_model_dir:
                self.init_pretrained_weights(self.variable_scope.name)

        if cache_len == 0:
            return output, None

        return output, new_memory
