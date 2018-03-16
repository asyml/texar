#
"""
Various utilities specific to data processing.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# pylint: disable=invalid-name

import tensorflow as tf

import numpy as np

from texar.core import utils

__all__ = [
    "random_shard_dataset",
    "count_file_lines"
]

def random_shard_dataset(dataset_size, shard_size, seed=None):
    """Returns a dataset transformation function that randomly shards a
    dataset.
    """
    num_shards = utils.ceildiv(dataset_size, shard_size)
    boundaries = np.linspace(
        0, dataset_size, num=num_shards, endpoint=False)

    def _shard_fn(dataset):
        sharded_dataset = (
            tf.data.Dataset.from_tensor_slices(boundaries)
            .shuffle(num_shards, seed=seed)
            .flat_map(lambda lb: dataset.skip(lb).take(shard_size)))
        return sharded_dataset

    return _shard_fn

def count_file_lines(filenames):
    """Counts the number of lines in the file(s).
    """
    def _count_lines(fn):
        with open(fn) as f:
            i = -1
            for i, _ in enumerate(f):
                pass
            return i + 1

    if not isinstance(filenames, (list, tuple)):
        filenames = [filenames]
    num_lines = np.sum([_count_lines(fn) for fn in filenames])
    return num_lines


