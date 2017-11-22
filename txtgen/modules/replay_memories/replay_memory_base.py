from txtgen.modules.module_base import ModuleBase

class ReplayMemoryBase(ModuleBase):
    def __init__(self, hparams=None):
        ModuleBase.__init__(self, hparams)

    def push(self, element):
        raise NotImplementedError

    def sample(self, size):
        raise NotImplementedError

    @staticmethod
    def default_hparams():
        return {
            'name': 'replay_memory'
        }