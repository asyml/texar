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

    def get_train_op(self, loss, variables=None, global_step=None,
                     increment_global_step=True):
        """Creates the train op of the model.

        Args:
            loss: A `Tensor` of the model loss.
            variables (list of Variables, optional): Variables to optimize. If
                `None`, the model variables are used.
            global_step (scalar int Tensor, optional): step counter to update
                on each step unless :attr:`increment_global_step` is `False`.
                If `None`, a new global step variable will be created.
            incremental_global_step (bool): Whether to increment
                :attr:`global_step`. This is useful if the :attr:`global_step`
                is used in multiple training ops per training step
                (e.g. to optimize different parts of the model) to avoid
                incrementing :attr:`global_step` more times than necessary.

        Returns:
            tuple: (train_op, global_step). If :attr:`global_step` is
            provided, the same :attr:`global_step` variable is returned,
            otherwise a new global step is created and returned.
        """
        return core.get_train_op(
            loss,
            variables=variables or self._trainable_variables,
            global_step=global_step,
            increment_global_step=increment_global_step,
            hparams=self._hparams)

