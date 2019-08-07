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
Base class for Pre-trained Modules.
"""

from texar.module_base import ModuleBase
from texar.modules.pretrained.bert_utils import (
    load_pretrained_bert, transform_bert_to_texar_config)
from texar.modules.pretrained.xlnet_utils import (
    load_pretrained_xlnet, transform_xlnet_to_texar_config)


__all__ = [
    "PretrainedBase",
]


class PretrainedBase(ModuleBase):
    r"""Base class for all pre-trained classes to inherit.

    Args:
        pretrained_model_name (optional): A str with the name
            of a pre-trained model to load. If `None`, will use the model
            name in :attr:`hparams`.
        cache_dir (optional): The path to a folder in which the
            pre-trained models will be cached. If `None` (default),
            a default directory will be used.
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparameter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter structure
            and default values.
    """

    def __init__(self,
                 pretrained_model_name=None,
                 cache_dir=None,
                 hparams=None):

        ModuleBase.__init__(self, hparams=hparams)

        self.pretrained_model_dir = None

        if self.model_name == "BERT":
            load_func = load_pretrained_bert
            transform_func = transform_bert_to_texar_config
        elif self.model_name == "XLNet":
            load_func = load_pretrained_xlnet
            transform_func = transform_xlnet_to_texar_config
        else:
            raise ValueError("Could not find this pre-trained model.")

        if pretrained_model_name:
            self.pretrained_model_dir = load_func(
                pretrained_model_name, cache_dir)
        elif self._hparams.pretrained_model_name is not None:
            self.pretrained_model_dir = load_func(
                self._hparams.pretrained_model_name, cache_dir)

        if self.pretrained_model_dir:
            self.pretrained_model_hparams = transform_func(
                self.pretrained_model_dir)

    @staticmethod
    def default_hparams():
        r"""Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                "pretrained_model_name": None,
                "name": "pretrained_base"
            }
        """
        return {
            'pretrained_model_name': None,
            'name': "pretrained_base",
            '@no_typecheck': ['pretrained_model_name']
        }

    def _build(self, inputs, *args, **kwargs):
        r"""Encodes the inputs and (optionally) conduct downstream prediction.

        Args:
            inputs: Inputs to the pre-trained module.
            *args: Other arguments.
            **kwargs: Keyword arguments.

        Returns:
            Encoding results or prediction results.
        """
        raise NotImplementedError
