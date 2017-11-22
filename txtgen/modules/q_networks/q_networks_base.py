from txtgen.modules.module_base import ModuleBase


class QNetworksBase(ModuleBase):
    def __init__(self, hparams=None):
        ModuleBase.__init__(self, hparams=hparams)

    @staticmethod
    def default_hparams():
        return {
            'name': 'q_network'
        }

    def train(self, mini_batch=None):
        raise NotImplementedError

    def get_qvalue(self, state_batch):
        raise NotImplementedError
