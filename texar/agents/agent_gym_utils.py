#
"""
Various agent utilities based on OpenAI Gym.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym

from texar.agents.agent_utils import Space

def convert_gym_space(spc):
    """Converts a gym Space instance to a
    :class:`~texar.agents.agent_utils.Space` instance.

    Args:
        spc: An instance of `gym.Space` or
            :class:`~texar.agents.agent_utils.Space`.
    """
    if isinstance(spc, Space):
        return spc
    if isinstance(spc, gym.spaces.Discrete):
        return Space(shape=(), low=0, high=spc.n, dtype=spc.dtype)
    elif isinstance(spc, gym.spaces.Box):
        return Space(
            shape=spc.shape, low=spc.low, high=spc.high, dtype=spc.dtype)

