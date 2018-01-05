#
"""
TODO: docs
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import deque
import random

from texar.hyperparams import HParams


class ReplayMemoryBase(object):
    """TODO: docs
    """
    def __init__(self, hparams=None):
        self._hparams = HParams(hparams, self.default_hparams())

    def add(self, element):
        """TODO: docs
        """
        raise NotImplementedError

    def get(self, size):
        """TODO: docs
        """
        raise NotImplementedError

    @staticmethod
    def default_hparams():
        """Returns a dictionary of default hyperparameters.
        """
        return {
            'name': 'replay_memory'
        }


class DequeReplayMemory(ReplayMemoryBase):
    """TODO: docs
    """
    def __init__(self, hparams=None):
        ReplayMemoryBase.__init__(self, hparams)
        self.deque = deque()
        self.capacity = self._hparams.capacity

    @staticmethod
    def default_hparams():
        return {
            'name': 'deque_replay_memory',
            'capacity': 80000
        }

    def add(self, element):
        self.deque.append(element)
        if len(self.deque) > self.capacity:
            self.deque.popleft()

    #TODO(zhiting): is it okay to have stand alone random generator ?
    def get(self, size):
        return random.sample(self.deque, size)
