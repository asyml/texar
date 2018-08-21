#
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

