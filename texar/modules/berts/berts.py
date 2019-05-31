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

from texar import ModuleBase
from texar.modules.berts import bert_utils

__all__ = [
    "BertBase",
]


class BertBase(ModuleBase):
    """Base class for all Bert classes to inherit.

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

    """

    def __init__(self,
                 pretrained_model_name=None,
                 cache_dir=None,
                 hparams=None):
        ModuleBase.__init__(self, hparams)

        if pretrained_model_name:
            self.pretrained_model = bert_utils.\
                load_pretrained_model(pretrained_model_name,
                                      cache_dir)
        elif self._hparams.pretrained_model_name is not None:
            self.pretrained_model = bert_utils.\
                load_pretrained_model(self._hparams.pretrained_model_name,
                                      cache_dir)
        else:
            self.pretrained_model = None

        if self.pretrained_model:
            self.pretrained_model_hparams = bert_utils. \
                transform_bert_to_texar_config(self.pretrained_model)

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                "name": "bert"
            }
        """
        return {
            'pretrained_model_name': 'bert-base-uncased',
            'name': 'bert_base',
            '@no_typecheck': ['pretrained_model_name']
        }

    def _build(self, inputs, *args, **kwargs):
        """Encodes the inputs and (optionally) conduct downstream prediction.

        Args:
            inputs: Inputs to the bert module.
            *args: Other arguments.
            **kwargs: Keyword arguments.

        Returns:
            Encoding results or prediction results.
        """
        raise NotImplementedError




