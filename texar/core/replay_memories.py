from texar.hyperparams import HParams

from collections import deque
import random


class ReplayMemoryBase:
    def __init__(self, hparams=None):
        self._hparams = HParams(hparams, self.default_hparams())

    def push(self, element):
        raise NotImplementedError

    def sample(self, size):
        raise NotImplementedError

    @staticmethod
    def default_hparams():
        return {
            'name': 'replay_memory'
        }


class DequeReplayMemory(ReplayMemoryBase):
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

    def push(self, element):
        self.deque.append(element)
        if len(self.deque) > self.capacity:
            self.deque.popleft()

    def sample(self, size):
        return random.sample(self.deque, size)
