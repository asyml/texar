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
Base class for reinforcement learning agents for sequence prediction.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from texar.agents.agent_base import AgentBase

# pylint: disable=too-many-instance-attributes

class SeqAgentBase(AgentBase):
    """
    Base class inherited by sequence prediction RL agents.

    Args:
        TODO
    """
    def __init__(self, hparams=None):
        AgentBase.__init__(self, hparams)


    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        TODO
        """
        return {
            'name': 'agent'
        }

