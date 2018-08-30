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
Various agent utilities based on OpenAI Gym.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym

__all__ = [
    "convert_gym_space",
    "get_gym_env_config"
]

def convert_gym_space(spc):
    """Converts a :gym:`gym.Space <#spaces>` instance to a
    :class:`~texar.agents.Space` instance.

    Args:
        spc: An instance of `gym.Space` or
            :class:`~texar.agents.Space`.
    """
    from texar.agents.agent_utils import Space
    if isinstance(spc, Space):
        return spc
    if isinstance(spc, gym.spaces.Discrete):
        return Space(shape=(), low=0, high=spc.n, dtype=spc.dtype)
    elif isinstance(spc, gym.spaces.Box):
        return Space(
            shape=spc.shape, low=spc.low, high=spc.high, dtype=spc.dtype)

def get_gym_env_config(env):
    """Creates an instance of :class:`~texar.agents.EnvConfig`
    from a :gym:`gym env <#environments>`.

    Args:
        env: An instance of OpenAI gym Environment.

    Returns:
        An instance of :class:`~texar.agents.EnvConfig`.
    """
    from texar.agents.agent_utils import EnvConfig
    return EnvConfig(
        action_space=env.action_space,
        observ_space=env.observation_space,
        reward_range=env.reward_range)

