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
Classes and utilities for replay memory in RL.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import deque
import random

from texar.hyperparams import HParams

__all__ = [
    "ReplayMemoryBase",
    "DequeReplayMemory"
]

class ReplayMemoryBase(object):
    """Base class of replay memory inheritted by all replay memory classes.

    Args:
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparameters are set to default values. See
            :meth:`default_hparams` for the defaults.
    """
    def __init__(self, hparams=None):
        self._hparams = HParams(hparams, self.default_hparams())

    @staticmethod
    def default_hparams():
        """Returns a `dict` of hyperparameters and their default values.

        .. code-block:: python

            {
                'name': 'replay_memory'
            }
        """
        return {
            'name': 'replay_memory'
        }

    def add(self, element):
        """Inserts a memory entry
        """
        raise NotImplementedError

    def get(self, size):
        """Pops a memory entry.
        """
        raise NotImplementedError

    def last(self):
        """Returns the latest element in the memeory.
        """
        raise NotImplementedError

    def size(self):
        """Returns the current size of the memory.
        """
        raise NotImplementedError


class DequeReplayMemory(ReplayMemoryBase):
    """A deque based replay memory that accepts new memory entry and deletes
    oldest memory entry if exceeding the capacity. Memory entries are
    accessed in random order.

    Args:
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparameters are set to default values. See
            :meth:`default_hparams` for the defaults.
    """
    def __init__(self, hparams=None):
        ReplayMemoryBase.__init__(self, hparams)
        self.deque = deque()
        self.capacity = self._hparams.capacity

    @staticmethod
    def default_hparams():
        """Returns a `dict` of hyperparameters and their default values.

        .. code-block:: python

            {
                'capacity': 80000,
                'name': 'deque_replay_memory',
            }

        Here:

        "capacity" : int
            Maximum size of memory kept. Deletes oldest memories if exceeds
            the capacity.
        """
        return {
            'name': 'deque_replay_memory',
            'capacity': 80000
        }

    def add(self, element):
        """Appends element to the memory and deletes old memory if exceeds
        the capacity.
        """
        self.deque.append(element)
        if len(self.deque) > self.capacity:
            self.deque.popleft()

    #TODO(zhiting): is it okay to have stand alone random generator ?
    def get(self, size):
        """Randomly samples :attr:`size` entries from the memory. Returns
        a list.
        """
        return random.sample(self.deque, size)

    def last(self):
        """Returns the latest element in the memeory.
        """
        return self.deque[-1]

    def size(self):
        """Returns the current size of the memory.
        """
        return len(self.deque)
