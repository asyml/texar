from texar.modules.module_base import ModuleBase


class QNetworkBase(ModuleBase):
    def __init__(self, hparams=None):
        ModuleBase.__init__(self, hparams=hparams)

    @staticmethod
    def default_hparams():
        return {
            'name': 'q_network'
        }
