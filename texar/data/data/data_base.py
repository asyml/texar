# Copyright 2018 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
    """Base class inheritted by all data classes.
    """

    def __init__(self, hparams):
        self._hparams = HParams(hparams, self.default_hparams())

    @staticmethod
    def default_hparams():
        """Returns a dictionary of default hyperparameters.

        .. code-block:: python

            {
                "num_epochs": 1,
                "batch_size": 64,
                "allow_smaller_final_batch": True,
                "shuffle": True,
                "shuffle_buffer_size": None,
                "shard_and_shuffle": False,
                "num_parallel_calls": 1,
                "prefetch_buffer_size": 0,
                "max_dataset_size": -1,
                "seed": None,
                "name": "data",
            }

        Here:

            "num_epochs" : int
                Number of times the dataset should be repeated. An
                :tf_main:`OutOfRangeError <errors/OutOfRangeError>` signal will
                be raised after the whole repeated dataset has been iterated
                through.

                E.g., For training data, set it to 1 (default) so that you
                will get the signal after each epoch of training. Set to -1
                to repeat the dataset indefinitely.

            "batch_size" : int
                Batch size, i.e., the number of consecutive elements of the
                dataset to combine in a single batch.

            "allow_smaller_final_batch" : bool
               Whether to allow the final batch to be smaller if there are
               insufficient elements left. If `False`, the final batch is
               discarded if it is smaller than batch size. Note that,
               if `True`, `output_shapes` of the resulting dataset
               will have a a **static** batch_size dimension equal to
               "batch_size".

            "shuffle" : bool
                Whether to randomly shuffle the elements of the dataset.

            "shuffle_buffer_size" : int
                The buffer size for data shuffling. The larger, the better
                the resulting data is mixed.

                If `None` (default), buffer size is set to the size of the
                whole dataset (i.e., make the shuffling the maximally
                effective).

            "shard_and_shuffle" : bool
                Whether to first shard the dataset and then shuffle each
                block respectively. Useful when the whole data is too large to
                be loaded efficiently into the memory.

                If `True`, :attr:`shuffle_buffer_size` must be specified to
                determine the size of each shard.

            "num_parallel_calls" : int
                Number of elements from the datasets to process in parallel.

            "prefetch_buffer_size" : int
                The maximum number of elements that will be buffered when
                prefetching.

            max_dataset_size : int
                Maximum number of instances to include in
                the dataset. If set to `-1` or greater than the size of
                dataset, all instances will be included. This constraint is
                imposed after data shuffling and filtering.

            seed : int, optional
                The random seed for shuffle.

                Note that if a seed is set, the shuffle order will be exact
                the same every time when going through the (repeated) dataset.

                For example, consider a dataset with elements [1, 2, 3], with
                "num_epochs"`=2` and some fixed seed, the resulting sequence
                can be: 2 1 3, 1 3 2 | 2 1 3, 1 3 2, ... That is, the orders are
                different **within** every `num_epochs`, but are the same
                **across** the `num_epochs`.

            name : str
                Name of the data.
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
    def _make_batch(dataset, hparams, padded_batch=False, padding_values=None):
        dataset = dataset.repeat(hparams.num_epochs)
        batch_size = hparams["batch_size"]
        if hparams["allow_smaller_final_batch"]:
            if padded_batch:
                dataset = dataset.padded_batch(
                    batch_size, dataset.output_shapes,
                    padding_values=padding_values)
            else:
                dataset = dataset.batch(batch_size)
        else:
            dataset = dataset.apply(
                tf.contrib.data.padded_batch_and_drop_remainder(
                    batch_size, dataset.output_shapes,
                    padding_values=padding_values))
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
        """A :class:`~texar.HParams` instance of the
        data hyperparameters.
        """
        return self._hparams

    @property
    def name(self):
        """Name of the module.
        """
        return self._hparams.name

