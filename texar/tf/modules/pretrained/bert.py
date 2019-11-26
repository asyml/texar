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
Utils of BERT Modules.
"""

import collections
import json
import os
import re

from abc import ABCMeta

import tensorflow as tf

from texar.tf.modules.pretrained.pretrained_base import PretrainedMixin

__all__ = [
    "PretrainedBERTMixin",
]

_BERT_PATH = "https://storage.googleapis.com/bert_models/"


class PretrainedBERTMixin(PretrainedMixin):
    r"""A mixin class to support loading pre-trained checkpoints for modules
    that implement the BERT model.

    The BERT model was proposed in (`Devlin et al`. 2018)
    `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding`_
    . A bidirectional Transformer language model pre-trained on large text
    corpora. Available model names include:

      * ``bert-base-uncased``: 12-layer, 768-hidden, 12-heads,
        110M parameters.
      * ``bert-large-uncased``: 24-layer, 1024-hidden, 16-heads,
        340M parameters.
      * ``bert-base-cased``: 12-layer, 768-hidden, 12-heads , 110M parameters.
      * ``bert-large-cased``: 24-layer, 1024-hidden, 16-heads,
        340M parameters.
      * ``bert-base-multilingual-uncased``: 102 languages, 12-layer,
        768-hidden, 12-heads, 110M parameters.
      * ``bert-base-multilingual-cased``: 104 languages, 12-layer, 768-hidden,
        12-heads, 110M parameters.
      * ``bert-base-chinese``: Chinese Simplified and Traditional, 12-layer,
        768-hidden, 12-heads, 110M parameters.

    We provide the following BERT classes:

      * :class:`~texar.tf.modules.BERTEncoder` for text encoding.
      * :class:`~texar.tf.modules.BERTClassifier` for text classification and
        sequence tagging.

    .. _`BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding`:
        https://arxiv.org/abs/1810.04805
    """

    __metaclass__ = ABCMeta

    _MODEL_NAME = "BERT"
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

    @classmethod
    def _transform_config(cls, pretrained_model_name, cache_dir):
        info = list(os.walk(cache_dir))
        root, _, files = info[0]
        config_path = None

        for file in files:
            if file.endswith('config.json'):
                config_path = os.path.join(root, file)
                with open(config_path) as f:
                    config_ckpt = json.loads(f.read())

        if config_path is None:
            raise ValueError("Cannot find the config file in {}".format(
                cache_dir))

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

    def _init_from_checkpoint(self, pretrained_model_name,
                              cache_dir, scope_name, **kwargs):
        tvars = tf.trainable_variables()
        init_checkpoint = os.path.abspath(os.path.join(cache_dir,
                                                       'bert_model.ckpt'))
        if init_checkpoint:
            assignment_map, initialized_variable_names = \
                self._get_assignment_map_from_checkpoint(
                    tvars, init_checkpoint, scope_name)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    def _get_assignment_map_from_checkpoint(self, tvars, init_checkpoint,
                                            scope_name):
        r"""`https://github.com/google-research/bert/blob/master/modeling.py`

        Compute the union of the current variables and checkpoint variables.
        Because the variable scope of the original BERT and Texar
        implementation, we need to build a assignment map to match the
        variables.
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
                check_name_scope = check_name.replace("bert/", scope_name + '/')
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
                                        'attention/self/output',
                                        check_name_scope)
                if model_name == check_name_scope:
                    model_name = check_name_scope.replace(
                        'attention/output/LayerNorm', 'output/LayerNorm')

                if model_name in name_to_variable.keys():
                    assignment_map[check_name] = model_name
                    initialized_variable_names[model_name] = 1
                    initialized_variable_names[model_name + ":0"] = 1
                else:
                    tf.logging.info(
                        'model name:{} not exist'.format(model_name))

        return assignment_map, initialized_variable_names
