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

import json
import os

from abc import ABC

from texar.tf.modules.pretrained.pretrained_base import PretrainedMixin

__all__ = [
    "PretrainedBERTMixin",
]

_BERT_PATH = "https://storage.googleapis.com/bert_models/"
_BIOBERT_PATH = "https://github.com/naver/biobert-pretrained/releases/download/"
_SCIBERT_PATH = "https://s3-us-west-2.amazonaws.com/ai2-s2-research/" \
                "scibert/tensorflow_models/"


class PretrainedBERTMixin(PretrainedMixin, ABC):
    r"""A mixin class to support loading pre-trained checkpoints for modules
    that implement the BERT model.

    Both standard BERT models and many domain specific BERT-based models are
    supported. You can specify the :attr:`pretrained_model_name` argument to
    pick which pre-trained BERT model to use. All available categories of
    pre-trained models (and names) include:

    * **Standard BERT**: proposed in (`Devlin et al`. 2018)
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

    * **BioBERT**: proposed in (`Lee et al`. 2019)
      `BioBERT: a pre-trained biomedical language representation model for biomedical text mining`_
      . A domain specific language representation model pre-trained on
      large-scale biomedical corpora. Based on the BERT architecture, BioBERT
      effectively transfers the knowledge from a large amount of biomedical
      texts to biomedical text mining models with minimal task-specific
      architecture modifications. Available model names include:

        * ``biobert-v1.0-pmc``: BioBERT v1.0 (+ PMC 270K) - based on
          BERT-base-Cased (same vocabulary).
        * ``biobert-v1.0-pubmed-pmc``: BioBERT v1.0 (+ PubMed 200K + PMC 270K) -
          based on BERT-base-Cased (same vocabulary).
        * ``biobert-v1.0-pubmed``: BioBERT v1.0 (+ PubMed 200K) - based on
          BERT-base-Cased (same vocabulary).
        * ``biobert-v1.1-pubmed``: BioBERT v1.1 (+ PubMed 1M) - based on
          BERT-base-Cased (same vocabulary).

    * **SciBERT**: proposed in (`Beltagy et al`. 2019)
      `SciBERT: A Pretrained Language Model for Scientific Text`_. A BERT model
      trained on scientific text. SciBERT leverages unsupervised pre-training
      on a large multi-domain corpus of scientific publications to improve
      performance on downstream scientific NLP tasks. Available model
      names include:

        * ``scibert-scivocab-uncased``: Uncased version of the model trained
          on its own vocabulary.
        * ``scibert-scivocab-cased``: Cased version of the model trained on
          its own vocabulary.
        * ``scibert-basevocab-uncased``: Uncased version of the model trained
          on the original BERT vocabulary.
        * ``scibert-basevocab-cased``: Cased version of the model trained on
          the original BERT vocabulary.

    We provide the following BERT classes:

      * :class:`~texar.torch.modules.BERTEncoder` for text encoding.
      * :class:`~texar.torch.modules.BERTClassifier` for text classification and
        sequence tagging.

    .. _`BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding`:
        https://arxiv.org/abs/1810.04805

    .. _`BioBERT: a pre-trained biomedical language representation model for biomedical text mining`:
        https://arxiv.org/abs/1901.08746

    .. _`SciBERT: A Pretrained Language Model for Scientific Text`:
        https://arxiv.org/abs/1903.10676
    """

    _MODEL_NAME = "BERT"
    _MODEL2URL = {
        # Standard BERT
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

        # BioBERT
        'biobert-v1.0-pmc':
            _BIOBERT_PATH + 'v1.0-pmc/biobert_v1.0_pmc.tar.gz',
        'biobert-v1.0-pubmed-pmc':
            _BIOBERT_PATH + 'v1.0-pubmed-pmc/biobert_v1.0_pubmed_pmc.tar.gz',
        'biobert-v1.0-pubmed':
            _BIOBERT_PATH + 'v1.0-pubmed/biobert_v1.0_pubmed.tar.gz',
        'biobert-v1.1-pubmed':
            _BIOBERT_PATH + 'v1.1-pubmed/biobert_v1.1_pubmed.tar.gz',

        # SciBERT
        'scibert-scivocab-uncased':
            _SCIBERT_PATH + 'scibert_scivocab_uncased.tar.gz',
        'scibert-scivocab-cased':
            _SCIBERT_PATH + 'scibert_scivocab_cased.tar.gz',
        'scibert-basevocab-uncased':
            _SCIBERT_PATH + 'scibert_basevocab_uncased.tar.gz',
        'scibert-basevocab-cased':
            _SCIBERT_PATH + 'scibert_basevocab_cased.tar.gz',
    }
    _MODEL2CKPT = {
        # Standard BERT
        'bert-base-uncased': 'bert_model.ckpt',
        'bert-large-uncased': 'bert_model.ckpt',
        'bert-base-cased': 'bert_model.ckpt',
        'bert-large-cased': 'bert_model.ckpt',
        'bert-base-multilingual-uncased': 'bert_model.ckpt',
        'bert-base-multilingual-cased': 'bert_model.ckpt',
        'bert-base-chinese': 'bert_model.ckpt',

        # BioBERT
        'biobert-v1.0-pmc': 'biobert_model.ckpt',
        'biobert-v1.0-pubmed-pmc': 'biobert_model.ckpt',
        'biobert-v1.0-pubmed': 'biobert_model.ckpt',
        'biobert-v1.1-pubmed': 'model.ckpt-1000000',

        # SciBERT
        'scibert-scivocab-uncased': 'bert_model.ckpt',
        'scibert-scivocab-cased': 'bert_model.ckpt',
        'scibert-basevocab-uncased': 'bert_model.ckpt',
        'scibert-basevocab-cased': 'bert_model.ckpt',
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
                              cache_dir, **kwargs):
        return
