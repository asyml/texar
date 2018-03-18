#
"""
Base class for models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from texar.module_base import ModuleBase
from texar import core

__all__ = [
    "ModelBase"
]

class ModelBase(ModuleBase):
    """Base class inherited by all model classes.
    """

    def __init__(self, hparams=None):
        ModuleBase.__init__(self, hparams)

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.
        """
        hparams = core.default_optimization_hparams()
        hparams.update({"name": "model"})
        return hparams

    def _build(self, *args, **kwargs):
        """The model logic.
        """
        raise NotImplementedError

    def get_loss(self):
        """Computes the loss of the model.
        """
        raise NotImplementedError

    def get_train_op(self):
        """Creates the train op of the model.
        """
        raise NotImplementedError

