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


def make_partial(fn, *args, **kwargs):
    """Returns a new function with single argument by freezing other arguments
    of :attr:`fn`.
    """
    def _new_fn(data):
        return fn(data, *args, **kwargs)
    return _new_fn

def make_chained_transformation(tran_fns, *args, **kwargs):
    """Returns a dataset transformation function that applies a list of
    transformations sequentially.

    Args:
        tran_fns (list): A list of dataset transformation.
        *args: Extra arguments for each of the transformation function.
        **kwargs: Extra keyword arguments for each of the transformation
            function.

    Returns:
        A transformation function to be used in
        :tf_main:`tf.data.Dataset.map <data/Dataset#map>`.
    """
    def _chained_fn(data):
        for tran_fns_i in tran_fns:
            data = tran_fns_i(data, *args, **kwargs)
        return data

    return _chained_fn

def make_combined_transformation(tran_fns, *args, **kwargs):
    """Returns a dataset transformation function that applies a list of
    transformations to each component of the data.

    The data to be transformed must be a tuple of the same length
    of :attr:`tran_fns`.

    Args:
        tran_fns (list): A list of elements where each element is a
            transformation function or a list of transformation functions.
        *args: Extra arguments for each of the transformation function.
        **kwargs: Extra keyword arguments for each of the transformation
            function.

    Returns:
        A transformation function to be used in
        :tf_main:`tf.data.Dataset.map <data/Dataset#map>`.
    """
    def _combined_fn(data):
        transformed_data = []
        for i, tran_fns_i in enumerate(tran_fns):
            data_i = data[i]
            if not isinstance(tran_fns_i, (list, tuple)):
                tran_fns_i = [tran_fns_i]
            for tran_fns_ij in tran_fns_i:
                data_i = tran_fns_ij(data_i, *args, **kwargs)
            transformed_data.append(data_i)
        return transformed_data

    return _combined_fn

def random_shard_dataset(dataset_size, shard_size, seed=None):
    """Returns a dataset transformation function that randomly shards a
    dataset.
    """
    num_shards = utils.ceildiv(dataset_size, shard_size)
    boundaries = np.linspace(0, dataset_size, num=num_shards, endpoint=False,
                             dtype=np.int64) #pylint: disable=no-member

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



