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

import tensorflow as tf

from texar.hyperparams import HParams
from texar.data.data import dataset_utils as dsutils
from texar.data.data_utils import count_file_lines

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

            max_dataset_size: int, maximum number of instances to include in
                the dataset. If set to `-1` or greater than the size of
                dataset, all instances will be included. This constraint is
                imposed after data shuffling and filtering.

        """
        return {
            "name": "data",
            "num_epochs": 1,
            "batch_size": 64,
            "allow_smaller_final_batch": True,
            "shuffle": True,
            "shuffle_buffer_size": None,
            "shard_and_shuffle": False,
            "num_parallel_calls": 1,
            "prefetch_buffer_size": 0,
            "max_dataset_size": -1,
            "seed": None
        }

    @staticmethod
    def _make_batch(dataset, hparams, padded_batch=False):
        dataset = dataset.repeat(hparams.num_epochs)
        batch_size = hparams["batch_size"]
        if hparams["allow_smaller_final_batch"]:
            if padded_batch:
                dataset = dataset.padded_batch(
                    batch_size, dataset.output_shapes)
            else:
                dataset = dataset.batch(batch_size)
        else:
            dataset = dataset.apply(
                tf.contrib.data.padded_batch_and_drop_remainder(
                    batch_size, dataset.output_shapes))
        return dataset

    @staticmethod
    def _shuffle_dataset(dataset, hparams, dataset_files):
        dataset_size = None
        shuffle_buffer_size = hparams["shuffle_buffer_size"]
        if hparams["shard_and_shuffle"]:
            if shuffle_buffer_size is None:
                raise ValueError(
                    "Dataset hyperparameter 'shuffle_buffer_size' "
                    "must not be `None` if 'shard_and_shuffle'=`True`.")
            dataset_size = count_file_lines(dataset_files)
            if shuffle_buffer_size >= dataset_size:
                raise ValueError(
                    "Dataset size (%d) <= shuffle_buffer_size (%d). Set "
                    "shuffle_and_shard to `False`." %
                    (dataset_size, shuffle_buffer_size))
            #TODO(zhiting): Use a different seed?
            dataset = dataset.apply(dsutils.random_shard_dataset(
                dataset_size, shuffle_buffer_size, hparams["seed"]))
            dataset = dataset.shuffle(shuffle_buffer_size + 16, # add a margin
                                      seed=hparams["seed"])
        elif hparams["shuffle"]:
            if shuffle_buffer_size is None:
                dataset_size = count_file_lines(dataset_files)
                shuffle_buffer_size = dataset_size
            dataset = dataset.shuffle(shuffle_buffer_size, seed=hparams["seed"])

        return dataset, dataset_size

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

