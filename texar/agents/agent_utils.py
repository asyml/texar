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
    """Observation and action spaces. Describes valid actions and observations.
    Similar to :gym:`gym.Space <#spaces>`.

    Args:
        shape (optional): Shape of the space, a tuple. If not
            given, infers from :attr:`low` and :attr:`high`.
        low (optional): Lower bound (inclusive) of each dimension of the
            space. Must have
            shape as specified by :attr:`shape`, and of the same shape with
            with :attr:`high` (if given). If `None`, set to `-inf` for each
            dimension.
        high (optional): Upper bound (inclusive) of each dimension of the
            space. Must have
            shape as specified by :attr:`shape`, and of the same shape with
            with :attr:`low` (if given). If `None`, set to `inf` for each
            dimension.
        dtype (optional): Data type of elements in the space. If not given,
            infers from :attr:`low` (if given) or set to `float`.

    Example:

        .. code-block:: python

            s = Space(low=0, high=10, dtype=np.int32)
            #s.contains(2) == True
            #s.contains(10) == True
            #s.contains(11) == False
            #s.shape == ()

            s2 = Space(shape=(2,2), high=np.ones([2,2]), dtype=np.float)
            #s2.low == [[-inf, -inf], [-inf, -inf]]
            #s2.high == [[1., 1.], [1., 1.]]
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
        else:
            shape = tuple(shape)

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
        self._shape = shape
        self._low = low
        self._high = high
        self._dtype = dtype

    def contains(self, x):
        """Checks if x is contained in the space. Returns a `bool`.
        """
        x = np.asarray(x)
        dtype_match = True
        if self._dtype.kind in np.typecodes['AllInteger']:
            if x.dtype.kind not in np.typecodes['AllInteger']:
                dtype_match = False
        shape_match = x.shape == self._shape
        low_match = (x >= self._low).all()
        high_match = (x <= self._high).all()
        return dtype_match and shape_match and low_match and high_match

    @property
    def shape(self):
        """Shape of the space.
        """
        return self._shape

    @property
    def low(self):
        """Lower bound of the space.
        """
        return self._low

    @property
    def high(self):
        """Upper bound of the space.
        """
        return self._high

    @property
    def dtype(self):
        """Data type of the element.
        """
        return self._dtype

class EnvConfig(object):
    """Configurations of an environment.

    Args:
        action_space: An instance of :class:`~texar.agents.Space` or
            :gym:`gym.Space <#spaces>`, the action space.
        observ_space: An instance of :class:`~texar.agents.Space` or
            :gym:`gym.Space <#spaces>`, the observation space.
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
