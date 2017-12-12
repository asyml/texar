#
"""TODO: docs
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from texar.modules.module_base import ModuleBase

class QNetBase(ModuleBase):
    """TODO: docs
    """
    def __init__(self, hparams=None):
        ModuleBase.__init__(self, hparams=hparams)

    @staticmethod
    def default_hparams():
        return {
            'name': 'q_net'
        }
