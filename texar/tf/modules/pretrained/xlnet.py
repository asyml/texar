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
Utils of XLNet Modules.
"""

import collections
import json
import os
import re

from abc import ABCMeta

import tensorflow as tf

from texar.tf.modules.pretrained.pretrained_base import PretrainedMixin

__all__ = [
    "PretrainedXLNetMixin",
]

_XLNET_PATH = "https://storage.googleapis.com/xlnet/released_models/"


class PretrainedXLNetMixin(PretrainedMixin):
    r"""A mixin class to support loading pre-trained checkpoints for modules
    that implement the XLNet model.

    The XLNet model was proposed in
    `XLNet: Generalized Autoregressive Pretraining for Language Understanding`_
    by `Yang et al.` It is based on the Transformer-XL model, pre-trained on a
    large corpus using a language modeling objective that considers all
    permutations of the input sentence.

    The available XLNet models are as follows:

      * ``xlnet-based-cased``: 12-layer, 768-hidden, 12-heads. This model is
        trained on full data (different from the one in the paper).
      * ``xlnet-large-cased``: 24-layer, 1024-hidden, 16-heads.

    We provide the following XLNet classes:

      * :class:`~texar.torch.modules.XLNetEncoder` for text encoding.
      * :class:`~texar.torch.modules.XLNetDecoder` for text generation and
        decoding.
      * :class:`~texar.torch.modules.XLNetClassifier` for text classification
        and sequence tagging.
      * :class:`~texar.torch.modules.XLNetRegressor` for text regression.

    .. _`XLNet: Generalized Autoregressive Pretraining for Language Understanding`:
        http://arxiv.org/abs/1906.08237
    """

    __metaclass__ = ABCMeta

    _MODEL_NAME = "XLNet"
    _MODEL2URL = {
        'xlnet-base-cased':
            _XLNET_PATH + "cased_L-12_H-768_A-12.zip",
        'xlnet-large-cased':
            _XLNET_PATH + "cased_L-24_H-1024_A-16.zip",
    }

    @classmethod
    def _transform_config(cls, pretrained_model_name, cache_dir):
        info = list(os.walk(cache_dir))
        root, _, files = info[0]
        config_path = None
        for file in files:
            if file.endswith('config.json'):
                config_path = os.path.join(root, file)
        if config_path is None:
            raise ValueError("Cannot find the config file in {}".format(
                cache_dir))

        with open(config_path) as f:
            config_ckpt = json.loads(f.read())

        configs = {
            "head_dim": config_ckpt["d_head"],
            "ffn_inner_dim": config_ckpt["d_inner"],
            "hidden_dim": config_ckpt["d_model"],
            "activation": config_ckpt["ff_activation"],
            "num_heads": config_ckpt["n_head"],
            "num_layers": config_ckpt["n_layer"],
            "vocab_size": config_ckpt["n_token"],
            "untie_r": config_ckpt["untie_r"]
        }

        return configs

    def _init_from_checkpoint(self, pretrained_model_name,
                              cache_dir, scope_name, **kwargs):

        tvars = tf.trainable_variables()
        init_checkpoint = os.path.join(cache_dir, 'xlnet_model.ckpt')
        if init_checkpoint:
            assignment_map, initialized_variable_names = \
                self._get_assignment_map_from_checkpoint(
                    tvars, init_checkpoint, scope_name)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    def _get_assignment_map_from_checkpoint(self, tvars, init_checkpoint,
                                            scope_name):
        r"""
        Compute the union of the current variables and checkpoint variables.
        Because of the variable scope of the original XLNet and Texar
        implementation, we need to build a assignment map to match the variables.
        """
        assignment_map = {}
        initialized_variable_names = {}

        name_to_variable = collections.OrderedDict()
        for var in tvars:
            name = var.name
            m = re.match("^(.*):\\d+$", name)
            if m is not None:
                name = m.group(1)
            name_to_variable[name] = var

        init_vars = tf.train.list_variables(init_checkpoint)

        for check_name, _ in init_vars:
            check_name_scope = check_name.replace(
                'model/transformer/', scope_name + '/')
            model_name = check_name_scope
            if check_name.startswith('model/lm_loss/bias'):
                model_name = scope_name + '/lm_loss/bias'
            elif check_name.startswith('model/transformer/mask_emb'):
                model_name = check_name_scope.replace(
                    'mask_emb/mask_emb', 'mask_emb')
            elif check_name.startswith('model/transformer/word_embedding'):
                model_name = scope_name + '/word_embedder/w'
            elif re.match('model/transformer/r_[r,s,w]_bias', check_name):
                model_name = check_name_scope
            elif re.match('model/transformer/seg_embed', check_name):
                model_name = check_name_scope
            elif re.match('model/transformer/layer_\\d+/rel_attn/[q,k,v,r,o]',
                          check_name):
                model_name = check_name_scope
            elif re.match('model/transformer/layer_\\d+/rel_attn/LayerNorm',
                          check_name):
                model_name = check_name_scope.replace('LayerNorm/', '')
            elif re.match('model/transformer/layer_\\d+/ff/layer_[1,2]',
                          check_name):
                model_name = check_name_scope.replace('ff/layer_1', 'ff/dense')
                if model_name == check_name_scope:
                    model_name = check_name_scope.replace(
                        'ff/layer_2', 'ff/dense_1')
            elif re.match('model/transformer/layer_\\d+/ff/LayerNorm',
                          check_name):
                model_name = check_name_scope.replace('LayerNorm/', '')

            if model_name in name_to_variable.keys():
                assignment_map[check_name] = model_name
                initialized_variable_names[model_name] = 1
                initialized_variable_names[model_name + ":0"] = 1
            else:
                tf.logging.info('model name:{} not exist'.format(model_name))

        return assignment_map, initialized_variable_names
