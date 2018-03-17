#
"""
Base data class that is enherited by all data classes.
A data defines data reading, parsing, batching, and other
preprocessing operations.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from texar.hyperparams import HParams

__all__ = [
    "DataBase"
]

class DataBase(object):
    """Base class of all text data classes.
    """

    def __init__(self, hparams):
        self._hparams = HParams(hparams, self.default_hparams())

    # TODO (zhiting): add more docs
    @staticmethod
    def default_hparams():
        """Returns a dictionary of default hyperparameters.
        """
        return {
            "name": "data",
            "num_epochs": None,
            "batch_size": 64,
            "allow_smaller_final_batch": True,
            "shuffle": True,
            "shuffle_buffer_size": None,
            "shard_and_shuffle": False,
            "num_parallel_calls": 1,
            "prefetch_buffer_size": 0,
            "seed": None
        }

    @property
    def num_epochs(self):
        """Number of epochs.
        """
        return self._hparams.num_epochs

    @property
    def batch_size(self):
        """The batch size.
        """
        return self._hparams.batch_size

    @property
    def hparams(self):
        """A :class:`~texar.hyperparams.HParams` instance of the
        data hyperparameters.
        """
        return self._hparams

    @property
    def name(self):
        """The data name.
        """
        return self._hparams.name

