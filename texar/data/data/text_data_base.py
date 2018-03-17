#
"""
Base text data class that is enherited by all text data classes.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from texar.data.data.data_base import DataBase


__all__ = [
    "TextDataBase"
]

class TextDataBase(DataBase): # pylint: disable=too-few-public-methods
    """Base class of all text data classes.
    """

    def __init__(self, hparams):
        DataBase.__init__(self, hparams)

    # TODO (zhiting): add more docs
    @staticmethod
    def default_hparams():
        """Returns a dictionary of default hyperparameters.
        """
        hparams = DataBase.default_hparams()
        hparams.update({
            "bucket_boundaries": [],
            "bucket_batch_sizes": None})
        return hparams

