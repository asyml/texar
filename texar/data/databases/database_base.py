#
"""
Base database class that is enherited by all database classes.
A database defines data reading, parsing, batching, and other
preprocessing operations.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from texar.hyperparams import HParams

__all__ = [
    "DataBaseBase"
]

class DataBaseBase(object):
    """Base class of all data classes.
    """

    def __init__(self, hparams):
        self._hparams = HParams(hparams, self.default_hparams())

    # TODO (zhiting): add more docs
    @staticmethod
    def default_hparams():
        """Returns a dictionary of default hyperparameters.
        """
        return {
            "name": "database",
            "num_epochs": None,
            "batch_size": 64,
            "bucket_batch_size": None,
            "allow_smaller_final_batch": False,
            "bucket_boundaries": [],
            "shuffle": True,
            "seed": None
        }

    @staticmethod
    def make_dataset(dataset_hparams):
        """Creates a Dataset instance that defines source filenames, data
        reading and decoding methods, vocabulary, and embedding initial
        values.

        Args:
            dataset_hparams (dict or HParams): Dataset hyperparameters.
        """
        raise NotImplementedError

    def _make_data_provider(self, dataset):
        """Creates a DataProvider instance that provides a single example of
        requested data.

        Args:
            dataset (Dataset): The dataset used to provide data examples.
        """
        raise NotImplementedError

    def __call__(self):
        """Returns a batch of data.
        """
        raise NotImplementedError

    @property
    def batch_size(self):
        """An integer representing the batch size.
        """
        return self._hparams.batch_size

    @property
    def hparams(self):
        """A :class:`~texar.hyperparams.HParams` instance of the
        database hyperparameters.
        """
        return self._hparams

    @property
    def name(self):
        """The name of the data base.
        """
        return self.hparams.name

