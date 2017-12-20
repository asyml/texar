"""
Basic class for QNets
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from texar.modules.module_base import ModuleBase

class QNetBase(ModuleBase):
    """
    Base class inherited by all Q-network.
    """
    def __init__(self, hparams=None):
        ModuleBase.__init__(self, hparams=hparams)

    @staticmethod
    def default_hparams():
        return {
            'name': 'q_net'
        }

    def _build(self, *args, **kwargs):
        raise NotImplementedError
