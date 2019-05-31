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
BERT encoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from texar.core import layers
from texar.modules.encoders.transformer_encoders import TransformerEncoder
from texar.modules.embedders import WordEmbedder, PositionEmbedder
from texar import HParams
from texar.modules.berts import BertBase, bert_utils
from texar.modules.encoders import EncoderBase

__all__ = [
    "BertEncoder",
]


class BertEncoder(BertBase, EncoderBase):
    """Raw BERT Transformer for encoding sequences.

    This module basically stacks
    :class:`~texar.modules.embedders.WordEmbedder`,
    :class:`~texar.modules.embedders.PositionEmbedder`,
    :class:`~texar.modules.encoders.TransformerEncoder` and a dense pooler.

    This module supports the architecture first proposed
    in `(Devlin et al.)` BERT.

    Args:
        pretrained_model_name (optional): a str with the name
            of a pre-trained model to load selected in the list of:
            `bert-base-uncased`, `bert-large-uncased`, `bert-base-cased`,
            `bert-large-cased`, `bert-base-multilingual-uncased`,
            `bert-base-multilingual-cased`, `bert-base-chinese`.
            If `None`, will use the model name in :attr:`hparams`.
        cache_dir (optional): the path to a folder in which the
            pre-trained models will be cached. If `None` (default),
            a default directory will be used.
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparameter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter sturcture
            and default values.

    .. document private functions
    .. automethod:: _build
    """
    def __init__(self,
                 pretrained_model_name=None,
                 cache_dir=None,
                 hparams=None):
        EncoderBase.__init__(self, hparams)
        BertBase.__init__(self, pretrained_model_name,
                          cache_dir, hparams)  # put these things to BertBase
        if self.pretrained_model:
            self._hparams = HParams(self.pretrained_model_hparams,
                                    self._hparams.todict())

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

            # The BERT encoder (a TransformerEncoder)
            self.encoder = TransformerEncoder(hparams=self._hparams.encoder)

            with tf.variable_scope("pooler"):
                kwargs_i = {"units": self._hparams.hidden_size,
                            "activation": tf.tanh}
                layer_hparams = {"type": "Dense", "kwargs": kwargs_i}
                self.pooler = layers.get_layer(hparams=layer_hparams)

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        * The encoder arch is determined by the constructor argument \
        :attr:`pretrained_model_name` if it's specified. In this case, \
        hparams are ignored.
        * Otherwise, the encoder arch is determined by \
        `hparams['pretrained_model_name']` if it's specified. All other \
        configs in hparams are ignored.
        * If the above two are `None`, the encoder arch is defined by \
        the configs in hparams and weights are randomly initialized.

        .. code-block:: python

            {
                'pretrained_model_name': 'bert-base-uncased',
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


        "pretrained_model_name" : str or None
             The name of the pretrained bert model. If None, the model
             will be randomly initialized.

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

        "hidden_size" : int
            Size of the pooler dense layer.

        "initializer" : dict, optional
            Hyperparameters of the default initializer that initializes
            variables created in this module.
            See :func:`~texar.core.get_initializer` for details.

        "name" : str
            Name of the module.
        """

        return {
            'pretrained_model_name': 'bert-base-uncased',
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
            'name': 'bert_encoder',
            '@no_typecheck': ['pretrained_model_name']
        }

    def _build(self,
               inputs,
               sequence_length=None,
               segment_ids=None,
               mode=None,
               **kwargs):
        """Encodes the inputs.

        Args:
            inputs: A 2D Tensor of shape `[batch_size, max_time]`,
                containing the token ids of tokens in the input sequences.
            segment_ids (optional): A 2D Tensor of shape
                `[batch_size, max_time]`, containing the segment ids
                of tokens in input sequences. If `None` (default), a
                tensor with all elements set to zero is used.
            sequence_length (optional): A 1D Tensor of shape `[batch_size]`.
                Input tokens beyond respective sequence lengths are masked
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
                `[batch_size, max_time, dim]` containing the \
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

        if sequence_length is None:
            sequence_length = tf.ones([batch_size], tf.int32) \
                              * tf.shape(inputs)[1]

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
