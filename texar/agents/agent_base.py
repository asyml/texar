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
Base class for reinforcement learning agents.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from texar.hyperparams import HParams
from texar.utils.variables import get_unique_named_variable_scope

# pylint: disable=too-many-instance-attributes

__all__ = [
    "AgentBase"
]

class AgentBase(object):
    """
    Base class inherited by RL agents.

    Args:
        TODO
    """
    def __init__(self, hparams=None):
        self._hparams = HParams(hparams, self.default_hparams())

        name = self._hparams.name
        self._variable_scope = get_unique_named_variable_scope(name)
        self._unique_name = self._variable_scope.name.split("/")[-1]

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        TODO
        """
        return {
            'name': 'agent'
        }

    @property
    def variable_scope(self):
        """The variable scope of the agent.
        """
        return self._variable_scope

    @property
    def name(self):
        """The name of the module (not uniquified).
        """
        return self._unique_name

    @property
    def hparams(self):
        """A :class:`~texar.hyperparams.HParams` instance. The hyperparameters
        of the module.
        """
        return self._hparams
