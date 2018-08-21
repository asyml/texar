# Copyright 2018 The Texar Authors. All Rights Reserved.
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
Base class for connectors that transform inputs into specified output shape.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from texar.module_base import ModuleBase

__all__ = [
    "ConnectorBase"
]

class ConnectorBase(ModuleBase):
    """Base class inherited by all connector classes. A connector is to
    transform inputs into outputs with any specified structure and shape.
    For example, tranforming the final state of an encoder to the initial
    state of a decoder, and performing stochastic sampling in between as
    in Variational Autoencoders (VAEs).

    Args:
        output_size: Size of output **excluding** the batch dimension. For
            example, set `output_size` to `dim` to generate output of
            shape `[batch_size, dim]`.
            Can be an `int`, a tuple of `int`, a Tensorshape, or a tuple of
            TensorShapes.
            For example, to transform inputs to have decoder state size, set
            `output_size=decoder.state_size`.
        hparams (dict, optional): Hyperparameters. Missing
            hyperparamerter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter sturcture and
            default values.
    """

    def __init__(self, output_size, hparams=None):
        ModuleBase.__init__(self, hparams)
        self._output_size = output_size

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.
        """
        return {
            "name": "connector"
        }

    def _build(self, *args, **kwargs):
        """Transforms inputs to outputs with specified shape.
        """
        raise NotImplementedError

    @property
    def output_size(self):
        """The output size.
        """
        return self._output_size
