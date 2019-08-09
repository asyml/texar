# Copyright 2019 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Utility functions related to XLNet encoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import os
import re

import tensorflow as tf
from texar.tf.modules.pretrained.pretrained_utils import default_download_dir
from texar.tf.data.data_utils import maybe_download

__all__ = [
    'init_xlnet_checkpoint',
    'load_pretrained_xlnet',
    'transform_xlnet_to_texar_config'
]

_XLNET_PATH = "https://storage.googleapis.com/xlnet/released_models/"
_MODEL2URL = {
    'xlnet-large-cased': _XLNET_PATH + "cased_L-24_H-1024_A-16.zip",
    'xlnet-base-cased': _XLNET_PATH + "cased_L-12_H-768_A-12.zip"
}


def _get_assignment_map_from_checkpoint(tvars,  # noqa: C901
                                        init_checkpoint, scope_name):
    """
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
        elif re.match('model/transformer/layer_\\d+/ff/LayerNorm', check_name):
            model_name = check_name_scope.replace('LayerNorm/', '')

        if model_name in name_to_variable.keys():
            assignment_map[check_name] = model_name
            initialized_variable_names[model_name] = 1
            initialized_variable_names[model_name + ":0"] = 1
        else:
            tf.logging.info('model name:{} not exist'.format(model_name))

    return assignment_map, initialized_variable_names


def init_xlnet_checkpoint(init_checkpoint_dir, scope_name):
    """
    Initialize XLnet model parameters from a checkpoint.

    Args:
        init_checkpoint_dir (str): path to the checkpoint.
        scope_name: variable scope of XLNet encoder.
    """
    tvars = tf.trainable_variables()
    init_checkpoint = os.path.join(init_checkpoint_dir, 'xlnet_model.ckpt')
    if init_checkpoint:
        assignment_map, initialized_variable_names = \
            _get_assignment_map_from_checkpoint(
                tvars, init_checkpoint, scope_name)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)


def load_pretrained_xlnet(pretrained_model_name, cache_dir=None):
    """
    Return the directory in which the pretrained model is cached.
    """
    if pretrained_model_name in _MODEL2URL:
        download_path = _MODEL2URL[pretrained_model_name]
    else:
        raise ValueError(
            "Pre-trained model not found: {}".format(pretrained_model_name))

    if cache_dir is None:
        cache_dir = default_download_dir("xlnet")

    file_name = download_path.split('/')[-1]
    # this is required because of the way xlnet model is bundled
    file_name = "xlnet_" + file_name

    cache_path = os.path.join(cache_dir, file_name.split('.')[0])
    if not os.path.exists(cache_path):
        maybe_download(download_path, cache_dir, extract=True)
    else:
        print("Using cached pre-trained model {} from: {}".format(
            pretrained_model_name, cache_dir))

    return cache_path


def transform_xlnet_to_texar_config(config_dir):
    """
    Load the Json config file and transform it into Texar style configuration.
    """
    config_ckpt = json.loads(
        open(os.path.join(config_dir, 'xlnet_config.json')).read())
    config = dict(untie_r=config_ckpt["untie_r"],
                  num_layers=config_ckpt["n_layer"],
                  # layer
                  head_dim=config_ckpt["d_head"],
                  hidden_dim=config_ckpt["d_model"],
                  num_heads=config_ckpt["n_head"],
                  vocab_size=config_ckpt["n_token"],
                  activation="gelu",
                  ffn_inner_dim=config_ckpt["d_inner"])

    return config
