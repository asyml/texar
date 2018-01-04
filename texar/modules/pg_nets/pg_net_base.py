"""
Base class for policy gradient networks
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from texar.modules import ModuleBase


class PGNetBase(ModuleBase):
    """
    Base class for policy gradient networks
    """
    def __init__(self, hparams=None):
        ModuleBase.__init__(self, hparams=hparams)

    @staticmethod
    def default_hparams():
        return {
            'name': 'pg_net_base'
        }

    def _build(self, *args, **kwargs):
        raise NotImplementedError
