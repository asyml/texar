from txtgen.modules.replay_memories.replay_memory_base import ReplayMemoryBase

from collections import deque
import random


class DequeReplayMemory(ReplayMemoryBase):
    def __init__(self, hparams=None):
        ReplayMemoryBase.__init__(self, hparams)
        self.deque = deque()
        self.max_size = hparams['max_size'] if hparams is not None else 80000

    def _build(self, *args, **kwargs):
        return

    @staticmethod
    def default_hparams():
        return {
            'name': 'deque_replay_memory',
            'max_size': 80000
        }

    def push(self, element):
        self.deque.append(element)
        if len(self.deque) > self.max_size:
            self.deque.popleft()

    def sample(self, size):
        return random.sample(self.deque, size)
