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
Various Bert modules.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from texar.core import layers
from texar.modules.encoders.transformer_encoders import TransformerEncoder
from texar.modules.embedders import WordEmbedder, PositionEmbedder
from texar import HParams, ModuleBase
from texar.utils import bert_utils

# pylint: disable=too-many-locals, invalid-name
# pylint: disable=arguments-differ, too-many-branches, too-many-statements

__all__ = [
    "BertBase",
    "BertEncoder",
    "BertForSequenceClassification"
]


class BertBase(ModuleBase):
    """Base class for all Bert classes to inherit.

    Args:
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparamerter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter sturcture and
            default values.
    """

    def __init__(self, hparams=None):
        ModuleBase.__init__(self, hparams)

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                "name": "bert"
            }
        """
        return {
            "name": "bert"
        }

    def _build(self, inputs, sequence_length, *args, **kwargs):
        """Encodes the inputs and (optionally) conduct downstream prediction.

        Args:
            inputs: Inputs to the bert module.
            sequence_length: Input tokens beyond respective
                sequence lengths are masked out automatically.
            *args: Other arguments.
            **kwargs: Keyword arguments.

        Returns:
            Encoding results or prediction results.
        """
        raise NotImplementedError


class BertEncoder(BertBase):
    """raw BERT Transformer for encoding sequences.

    This module basically stacks
    :class:`~texar.modules.embedders.WordEmbedder`,
    :class:`~texar.modules.embedders.PositionEmbedder`,
    :class:`~texar.modules.encoders.TransformerEncoder` and a dense pooler.

    This module supports the architecture first proposed
    in `(Devlin et al.)` BERT.

    Args:
        pretrained_model_name (optional): a str with the name
            of a pre-trained model
        to load selected in the list of:
                    . `bert-base-uncased`
                    . `bert-large-uncased`
                    . `bert-base-cased`
                    . `bert-large-cased`
                    . `bert-base-multilingual-uncased`
                    . `bert-base-multilingual-cased`
                    . `bert-base-chinese`
        cache_dir (optional): the path to a folder in which the
            pre-trained models will be cached.
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparameter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter sturcture and
            default values.

    .. document private functions
    .. automethod:: _build
    """
    def __init__(self,
                 pretrained_model_name=None,
                 cache_dir=None,
                 hparams=None):

        if pretrained_model_name is None:
            BertBase.__init__(self, hparams)
            self.pretrained_model = None
        else:
            self.pretrained_model = bert_utils.load_pretrained_model(pretrained_model_name, cache_dir)

            pretrained_model_params = bert_utils.transform_bert_to_texar_config(self.pretrained_model)
            BertBase.__init__(self, HParams(hparams, default_hparams=pretrained_model_params))

        with tf.variable_scope(self.variable_scope):
            if self._hparams.initializer:
                tf.get_variable_scope().set_initializer(
                    layers.get_initializer(self._hparams.initializer))

            # Word embedding
            self.word_embedder = WordEmbedder(
                vocab_size=self._hparams.vocab_size,
                hparams=self._hparams.embed)

            # Segment embedding for each type of tokens
            self.segment_embedder = WordEmbedder(
                vocab_size=self._hparams.type_vocab_size,
                hparams=self._hparams.segment_embed)

            # Position embedding
            self.position_embedder = PositionEmbedder(
                position_size=self._hparams.position_size,
                hparams=self._hparams.position_embed)

            # The BERT model (a TransformerEncoder)
            self.encoder = TransformerEncoder(hparams=self._hparams.encoder)

            with tf.variable_scope("pooler"):
                kwargs_i = {"units": self._hparams.hidden_size,
                            "activation": tf.tanh}
                layer_hparams = {"type": "Dense", "kwargs": kwargs_i}
                self.pooler = layers.get_layer(hparams=layer_hparams)

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                'embed': {
                    'dim': 768,
                    'name': 'word_embeddings'
                },
                'vocab_size': 30522,
                'segment_embed': {
                    'dim': 768,
                    'name': 'token_type_embeddings'
                },
                'type_vocab_size': 2,
                'position_embed': {
                    'dim': 768,
                    'name': 'position_embeddings'
                },
                'position_size': 512,

                'encoder': {
                    'dim': 768,
                    'embedding_dropout': 0.1,
                    'multihead_attention': {
                        'dropout_rate': 0.1,
                        'name': 'self',
                        'num_heads': 12,
                        'num_units': 768,
                        'output_dim': 768,
                        'use_bias': True
                    },
                    'name': 'encoder',
                    'num_blocks': 12,
                    'poswise_feedforward': {
                        'layers': [
                            {   'kwargs': {
                                    'activation': 'gelu',
                                    'name': 'intermediate',
                                    'units': 3072,
                                    'use_bias': True
                                },
                                'type': 'Dense'
                            },
                            {   'kwargs': {'activation': None,
                                'name': 'output',
                                'units': 768,
                                'use_bias': True
                                },
                                'type': 'Dense'
                            }
                        ]
                    },
                    'residual_dropout': 0.1,
                    'use_bert_config': True
                },
                'hidden_size': 768,
                'initializer': None,
                'name': 'bert_encoder'
            }



        Here:

        The default parameters are values for uncased BERT-Base model.

        "embed" : dict
            Hyperparameters for word embedding layer.

        "vocab_size" : int
            The vocabulary size of `inputs` in `BertModel`.

        "segment_embed" : dict
            Hyperparameters for segment embedding layer.

        "type_vocab_size" : int
            The vocabulary size of the `segment_ids` passed into `BertModel`.

        "position_embed" : dict
            Hyperparameters for position embedding layer.

        "position_size" :  int
            The maximum sequence length that this model might ever be used with.

        "encoder" : dict
            Hyperparameters for the TransformerEncoder.
            See :func:`~texar.modules.TransformerEncoder.default_harams`
            for details.

        "hidden_size" :
            Size of the encoder layers and the pooler layer.

        "initializer" : dict, optional
            Hyperparameters of the default initializer that initializes
            variables created in this module.
            See :func:`~texar.core.get_initializer` for details.

        "name" : str
            Name of the module.
        """

        return {
            'embed': {
                'dim': 768,
                'name': 'word_embeddings'
            },
            'vocab_size': 30522,
            'segment_embed': {
                'dim': 768,
                'name': 'token_type_embeddings'
            },
            'type_vocab_size': 2,
            'position_embed': {
                'dim': 768,
                'name': 'position_embeddings'
            },
            'position_size': 512,

            'encoder': {
                'dim': 768,
                'embedding_dropout': 0.1,
                'multihead_attention': {
                    'dropout_rate': 0.1,
                    'name': 'self',
                    'num_heads': 12,
                    'num_units': 768,
                    'output_dim': 768,
                    'use_bias': True
                },
                'name': 'encoder',
                'num_blocks': 12,
                'poswise_feedforward': {
                    'layers': [
                        {
                            'kwargs': {
                                'activation': 'gelu',
                                'name': 'intermediate',
                                'units': 3072,
                                'use_bias': True
                            },
                            'type': 'Dense'
                        },
                        {
                            'kwargs': {
                                'activation': None,
                                'name': 'output',
                                'units': 768,
                                'use_bias': True
                            },
                            'type': 'Dense'
                        }
                    ]
                },
                'residual_dropout': 0.1,
                'use_bert_config': True
            },
            'hidden_size': 768,
            'initializer': None,
            'name': 'bert_encoder'
        }

    def _build(self,
               inputs,
               sequence_length,
               segment_ids=None,
               mode=None, **kwargs):
        """Encodes the inputs.

        Args:
            inputs: A 2D Tensor of shape `[batch_size, max_time]`,
                containing the token ids of tokens in input sequences.
            segment_ids (optional): A 2D Tensor of shape
                `[batch_size, max_time], containing the segment ids
                of tokens in input sequences. If `None` (default), a
                tensor with all elements set to zero is used.
            sequence_length: A 1D Tensor of shape `[batch_size]`. Input
                tokens beyond respective sequence lengths are masked
                out automatically.
            mode (optional): A tensor taking value in
                :tf_main:`tf.estimator.ModeKeys <estimator/ModeKeys>`,
                including `TRAIN`, `EVAL`, and `PREDICT`. Used to toggle
                dropout.
                If `None` (default), :func:`texar.global_mode` is used.
            **kwargs: Keyword arguments.

        Returns:
            A pair :attr:`(outputs, pooled_output)`

                - :attr:`outputs`:  A Tensor of shape \
                `[batch_size, max_time, hidden_size]` containing the \
                 encoded vectors.

                - :attr:`pooled_output`: A Tensor of size \
                `[batch_size, hidden_size]` which is the output of a \
                pooler pretrained on top of the hidden state associated \
                to the first character of the input (`CLS`), see BERT's \
                paper.
        """

        if segment_ids is None:
            segment_ids = tf.zeros_like(inputs)

        word_embeds = self.word_embedder(inputs)

        segment_embeds = self.segment_embedder(segment_ids)

        batch_size = tf.shape(inputs)[0]
        pos_length = tf.ones([batch_size], tf.int32) * tf.shape(inputs)[1]
        pos_embeds = self.position_embedder(sequence_length=pos_length)

        input_embeds = word_embeds + segment_embeds + pos_embeds

        output = self.encoder(input_embeds, sequence_length, mode)

        with tf.variable_scope("pooler"):
            # taking the hidden state corresponding to the first token.
            first_token_tensor = tf.squeeze(output[:, 0:1, :], axis=1)
            pooled_output = self.pooler(first_token_tensor)

        if not self._built:
            self._add_internal_trainable_variables()
            self._built = True

            if self.pretrained_model:
                bert_utils.init_bert_checkpoint(self.pretrained_model,
                                                self.variable_scope.name)

        return output, pooled_output


class BertForSequenceClassification(BertBase):
    """BERT model for sequence classification.

    This module basically stacks
    :class:`~texar.modules.BertEncoder` with a linear layer on top of
    the pooled output.

    This module supports the architecture first proposed in
    `(Devlin et al.)` BERT.

    Args:
        pretrained_model_name (optional): a str with the name of a
        pre-trained model
        to load selected in the list of:
                    . `bert-base-uncased`
                    . `bert-large-uncased`
                    . `bert-base-cased`
                    . `bert-large-cased`
                    . `bert-base-multilingual-uncased`
                    . `bert-base-multilingual-cased`
                    . `bert-base-chinese`
        cache_dir (optional): the path to a folder in which the
            pre-trained models will be cached.
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparameter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter sturcture and
            default values.

    .. document private functions
    .. automethod:: _build
    """

    def __init__(self,
                 pretrained_model_name=None,
                 cache_dir=None,
                 hparams=None):

        BertBase.__init__(self, hparams)

        with tf.variable_scope(self.variable_scope):
            if self._hparams.initializer:
                tf.get_variable_scope().set_initializer(
                    layers.get_initializer(self._hparams.initializer))

            self.bert = BertEncoder(pretrained_model_name,
                                    cache_dir, self._hparams.bert_hparam)

            with tf.variable_scope("sequence_classification"):
                kwargs_i = {"rate": self._hparams.dropout}
                layer_hparams = {"type": "Dropout", "kwargs": kwargs_i}
                self.dropout = layers.get_layer(hparams=layer_hparams)

                kwargs_i = {"units": self._hparams.class_num}
                layer_hparams = {"type": "Dense", "kwargs": kwargs_i}
                self.classifier = layers.get_layer(hparams=layer_hparams)

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                "bert_hparam": None,
                "dropout": 0.1,
                "class_num": 2,
                "initializer": None,
                "name": "bert_for_sequence_classification"
            }


        Here:

        "bert_hparam" : dict
            Hyperparameters for the raw BERT Encoder.
            See :func:`~texar.modules.BertEncoder.default_harams`
            for details.

        "dropout" : float
            The dropout rate for the pooled output.

        "class_num" : int
            The number of classes for the classifier.

        "initializer" : dict, optional
            Hyperparameters of the default initializer that initializes
            variables created in this module.
            See :func:`~texar.core.get_initializer` for details.

        "name" : str
            Name of the module.
        """

        return {
            "bert_hparam": None,
            "dropout": 0.1,
            "class_num": 2,
            "initializer": None,
            "name": "bert_for_sequence_classification"
        }

    def _build(self,
               inputs,
               sequence_length,
               segment_ids=None,
               mode=None,
               **kwargs):
        """Compute the classification logits.

        Args:
            inputs: A 2D Tensor of shape `[batch_size, max_time]`,
                containing the token ids of tokens in input sequences.
            sequence_length: A 1D Tensor of shape `[batch_size]`. Input tokens
                beyond respective sequence lengths are masked out automatically.
            segment_ids (optional): A 2D Tensor of shape
                `[batch_size, max_time]`, containing the segment ids
                of tokens in input sequences. If `None` (default), a tensor
                 with all elements set to zero is used.
            mode (optional): A tensor taking value in
                :tf_main:`tf.estimator.ModeKeys <estimator/ModeKeys>`,
                including `TRAIN`, `EVAL`, and `PREDICT`. Used to toggle
                dropout.
                If `None` (default), :func:`texar.global_mode` is used.
            **kwargs: Keyword arguments.

        Returns:
            Outputs the classification logits of shape
             `[batch_size, num_labels]`.
        """

        _, pooled_output = self.bert(inputs,
                                     sequence_length, segment_ids, mode)

        with tf.variable_scope("sequence_classification"):
            pooled_output = self.dropout(pooled_output, training=mode)
            logits = self.classifier(pooled_output)

        if not self._built:
            self._add_internal_trainable_variables()
            self._built = True

        return logits
