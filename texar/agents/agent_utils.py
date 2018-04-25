#
"""
Various agent utilities.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=too-many-arguments, too-few-public-methods, no-member
# pylint: disable=invalid-name, wrong-import-position

import numpy as np

gym_utils = None
try:
    from texar.agents import agent_gym_utils as gym_utils
except ImportError:
    pass

__all__ = [
    "Space",
    "EnvConfig"
]

class Space(object):
    """Observation and action spaces. Similar to gym Space.
    """
    def __init__(self, shape=None, low=None, high=None, dtype=None):
        if low is None:
            low = -float('inf')
        if high is None:
            high = float('inf')
        if shape is None:
            low = np.asarray(low)
            high = np.asarray(high)
            if low.shape != high.shape:
                raise ValueError('`low` and `high` must have the same shape.')
            shape = low.shape
        if np.isscalar(low):
            low = low + np.zeros(shape, dtype=dtype)
        if np.isscalar(high):
            high = high + np.zeros(shape, dtype=dtype)
        if shape != low.shape or shape != high.shape:
            raise ValueError(
                'Shape inconsistent: shape={}, low.shape={}, high.shape={}'
                .format(shape, low.shape, high.shape))
        if dtype is None:
            dtype = low.dtype
        dtype = np.dtype(dtype)
        low = low.astype(dtype)
        high = high.astype(dtype)
        self.shape = shape
        self.low = low
        self.high = high
        self.dtype = dtype

    def contains(self, x):
        """Checks if :attr:`x` is contained in the space.
        """
        x = np.asarray(x)
        dtype_match = True
        if self.dtype.kind in np.typecodes['AllInteger']:
            if x.dtype.kind not in np.typecodes['AllInteger']:
                dtype_match = False
        shape_match = x.shape == self.shape
        low_match = (x >= self.low).all()
        high_match = (x <= self.high).all()
        return dtype_match and shape_match and low_match and high_match

class EnvConfig(object):
    """Configurations of environment.

    Args:
        action_space: An instance of `gym.Space` or
            :class:`~texar.agents.agent_utils.Space`.
        observ_space: An instance of `gym.Space` or
            :class:`~texar.agents.agent_utils.Space`.
        reward_range: A tuple corresponding to the min and max possible
            rewards, e.g., `reward_range=(-1.0, 1.0)`.
    """

    def __init__(self,
                 action_space,
                 observ_space,
                 reward_range):
        if gym_utils:
            action_space = gym_utils.convert_gym_space(action_space)
            observ_space = gym_utils.convert_gym_space(observ_space)

        self.action_space = action_space
        self.action_dtype = action_space.dtype
        self.action_shape = action_space.shape

        self.observ_space = observ_space
        self.observ_dtype = observ_space.dtype
        self.observ_shape = observ_space.shape

        self.reward_range = reward_range
