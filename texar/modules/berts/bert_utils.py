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
Utility functions related to BERT encoders.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import json
import collections
import re
import sys
import os
import tensorflow as tf
from texar.data.data_utils import maybe_download
__all__ = [
    "transform_bert_to_texar_config",
    "init_bert_checkpoint",
    "load_pretrained_model"
]

_BERT_PATH = "https://storage.googleapis.com/bert_models/"
_MODEL2URL = {
    'bert-base-uncased':
        _BERT_PATH + "2018_10_18/uncased_L-12_H-768_A-12.zip",
    'bert-large-uncased':
        _BERT_PATH + "2018_10_18/uncased_L-24_H-1024_A-16.zip",
    'bert-base-cased':
        _BERT_PATH + "2018_10_18/cased_L-12_H-768_A-12.zip",
    'bert-large-cased':
        _BERT_PATH + "2018_10_18/cased_L-24_H-1024_A-16.zip",
    'bert-base-multilingual-uncased':
        _BERT_PATH + "2018_11_23/multi_cased_L-12_H-768_A-12.zip",
    'bert-base-multilingual-cased':
        _BERT_PATH + "2018_11_03/multilingual_L-12_H-768_A-12.zip",
    'bert-base-chinese':
        _BERT_PATH + "2018_11_03/chinese_L-12_H-768_A-12.zip",
}

def _get_assignment_map_from_checkpoint(tvars, init_checkpoint, scope_name):
    """
    Provided by Google AI Language Team.
    Compute the union of the current variables and checkpoint variables.
    Because the variable scope of the original BERT and Texar implementation,
    we need to build a assignment map to match the variables.
    """
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = {
        'bert/embeddings/word_embeddings':
            scope_name + '/word_embeddings/w',
        'bert/embeddings/token_type_embeddings':
            scope_name + '/token_type_embeddings/w',
        'bert/embeddings/position_embeddings':
            scope_name + '/position_embeddings/w',
        'bert/embeddings/LayerNorm/beta':
            scope_name + '/encoder/LayerNorm/beta',
        'bert/embeddings/LayerNorm/gamma':
            scope_name + '/encoder/LayerNorm/gamma',
    }
    for check_name, model_name in assignment_map.items():
        initialized_variable_names[model_name] = 1
        initialized_variable_names[model_name + ":0"] = 1

    for check_name, _ in init_vars:
        if check_name.startswith('bert'):
            if check_name.startswith('bert/embeddings'):
                continue
            check_name_scope = check_name.replace("bert/", scope_name+'/')
            model_name = re.sub(
                'layer_\\d+/output/dense',
                lambda x: x.group(0).replace('output/dense', 'ffn/output'),
                check_name_scope)
            if model_name == check_name_scope:
                model_name = re.sub(
                    'layer_\\d+/output/LayerNorm',
                    lambda x: x.group(0).replace('output/LayerNorm',
                                                 'ffn/LayerNorm'),
                    check_name_scope)
            if model_name == check_name_scope:
                model_name = re.sub(
                    'layer_\\d+/intermediate/dense',
                    lambda x: x.group(0).replace('intermediate/dense',
                                                 'ffn/intermediate'),
                    check_name_scope)
            if model_name == check_name_scope:
                model_name = re.sub('attention/output/dense',
                                    'attention/self/output', check_name_scope)
            if model_name == check_name_scope:
                model_name = check_name_scope.replace(
                    'attention/output/LayerNorm', 'output/LayerNorm')

            if model_name in name_to_variable.keys():
                assignment_map[check_name] = model_name
                initialized_variable_names[model_name] = 1
                initialized_variable_names[model_name + ":0"] = 1
            else:
                tf.logging.info('model name:{} not exist'.format(model_name))

    return assignment_map, initialized_variable_names


def init_bert_checkpoint(init_checkpoint_dir, scope_name):
    """
    Initializes BERT model parameters from a checkpoint.
    Provided by Google AI Language Team.

    Args:
        init_checkpoint_dir (str): path to the checkpoint.
        scope_name: variable scope of bert encoder.
    """
    tvars = tf.trainable_variables()
    init_checkpoint = os.path.join(init_checkpoint_dir, 'bert_model.ckpt')
    if init_checkpoint:
        (assignment_map, initialized_variable_names
        ) = _get_assignment_map_from_checkpoint(
            tvars, init_checkpoint, scope_name)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)


def _default_download_dir():
    """
    Return the directory to which packages will be downloaded by default.
    """
    package_dir = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(__file__))))
    if os.access(package_dir, os.W_OK):
        texar_download_dir = os.path.join(package_dir, 'texar_download')
    else:
        # On Windows, use %APPDATA%
        if sys.platform == 'win32' and 'APPDATA' in os.environ:
            home_dir = os.environ['APPDATA']

        # Otherwise, install in the user's home directory.
        else:
            home_dir = os.path.expanduser('~/')
            if home_dir == '~/':
                raise ValueError("Could not find a default download directory")

        texar_download_dir = os.path.join(home_dir, 'texar_download')

    if not os.path.exists(texar_download_dir):
        os.mkdir(texar_download_dir)

    return os.path.join(texar_download_dir, 'bert')


def load_pretrained_model(pretrained_model_name, cache_dir):
    """
    Return the directory in which the pretrained model is cached.
    """
    if pretrained_model_name in _MODEL2URL:
        download_path = _MODEL2URL[pretrained_model_name]
    else:
        raise ValueError(
            "Pre-trained model not found: {}".format(pretrained_model_name))

    if cache_dir is None:
        cache_dir = _default_download_dir()

    file_name = download_path.split('/')[-1]

    cache_path = os.path.join(cache_dir, file_name.split('.')[0])
    if not os.path.exists(cache_path):
        maybe_download(download_path, cache_dir, extract=True)
    else:
        print("Using cached pre-trained BERT model from: %s." % cache_path)

    return cache_path


def transform_bert_to_texar_config(config_dir):
    """
    Load the Json config file and transform it into Texar style configuration.
    """
    config_ckpt = json.loads(
        open(os.path.join(config_dir, 'bert_config.json')).read())
    configs = {}
    hidden_dim = config_ckpt['hidden_size']
    configs['hidden_size'] = config_ckpt['hidden_size']
    configs['embed'] = {
        'name': 'word_embeddings',
        'dim': hidden_dim}
    configs['vocab_size'] = config_ckpt['vocab_size']

    configs['segment_embed'] = {
        'name': 'token_type_embeddings',
        'dim': hidden_dim}
    configs['type_vocab_size'] = config_ckpt['type_vocab_size']

    configs['position_embed'] = {
        'name': 'position_embeddings',
        'dim': hidden_dim}
    configs['position_size'] = config_ckpt['max_position_embeddings']

    configs['encoder'] = {
        'name': 'encoder',
        'embedding_dropout': config_ckpt['hidden_dropout_prob'],
        'num_blocks': config_ckpt['num_hidden_layers'],
        'multihead_attention': {
            'use_bias': True,
            'num_units': hidden_dim,
            'num_heads': config_ckpt['num_attention_heads'],
            'output_dim': hidden_dim,
            'dropout_rate': config_ckpt['attention_probs_dropout_prob'],
            'name': 'self'
        },
        'residual_dropout': config_ckpt['hidden_dropout_prob'],
        'dim': hidden_dim,
        'use_bert_config': True,
        'poswise_feedforward': {
            "layers": [
                {
                    'type': 'Dense',
                    'kwargs': {
                        'name': 'intermediate',
                        'units': config_ckpt['intermediate_size'],
                        'activation': config_ckpt['hidden_act'],
                        'use_bias': True,
                    }
                },
                {
                    'type': 'Dense',
                    'kwargs': {
                        'name': 'output',
                        'units': hidden_dim,
                        'activation': None,
                        'use_bias': True,
                    }
                },
            ],
        },
    }
    return configs
