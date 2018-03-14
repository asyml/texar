#
"""
Base class for encoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from texar.modules.module_base import ModuleBase

__all__ = [
    "ClassifierBase"
]

class ClassifierBase(ModuleBase):
    """Base class inherited by all classifier classes.
    """

    def __init__(self, hparams=None):
        ModuleBase.__init__(self, hparams)

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.
        """
        return {
            "name": "classifier"
        }

    def _build(self, inputs, *args, **kwargs):
        """Classifies the inputs.

        Args:
          inputs: Inputs to the classifier.
          *args: Other arguments.
          **kwargs: Keyword arguments.

        Returns:
          Classification results.
        """
        raise NotImplementedError

