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
Various convolutional network encoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from texar.modules.encoders.encoder_base import EncoderBase
from texar.modules.networks.conv_networks import Conv1DNetwork

__all__ = [
    "Conv1DEncoder"
]

class Conv1DEncoder(Conv1DNetwork, EncoderBase):
    """Simple Conv-1D encoder which consists of a sequence of conv layers
    followed with a sequence of dense layers.

    Wraps :class:`~texar.modules.Conv1DNetwork` to be a subclass of
    :class:`~texar.modules.EncoderBase`. Has exact the same functionality
    with :class:`~texar.modules.Conv1DNetwork`.
    """

    def __init__(self, hparams=None): # pylint: disable=super-init-not-called
        Conv1DNetwork.__init__(self, hparams)

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        The same as :meth:`~texar.modules.Conv1DNetwork.default_hparams`
        of :class:`~texar.modules.Conv1DNetwork`, except that the default name
        is 'conv_encoder'.
        """
        hparams = Conv1DNetwork.default_hparams()
        hparams['name'] = 'conv_encoder'
        return hparams

