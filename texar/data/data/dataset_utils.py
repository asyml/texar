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
Various utilities specific to dataset processing.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import six

import tensorflow as tf

import numpy as np

from texar.utils import utils

# pylint: disable=invalid-name, too-many-arguments

__all__ = [
    "_DataSpec",
    "_connect_name",
    "maybe_tuple",
    "make_partial",
    "make_chained_transformation",
    "make_combined_transformation",
    "random_shard_dataset",
]

class _DataSpec(object):
    """Dataset specification. Used to pass necessary info to
    user-defined tranformation functions.

    Args:
        dataset: Instance of :tf_main:`tf.data.Dataset <data/Dataset>`.
        dataset_size (int): Number of data samples.
        decoder: A (list of) data decoder.
        vocab: A (list of) :class:`~texar.data.Vocab` instance.
        embeddidng: A (list of) :class:`~texar.data.Embedding` instance.
        **kwargs: Any remaining dataset-specific fields.
    """
    def __init__(self, dataset=None, dataset_size=None, decoder=None,
                 vocab=None, embedding=None, **kwargs):
        kwargs['dataset'] = dataset
        kwargs['dataset_size'] = dataset_size
        kwargs['decoder'] = decoder
        kwargs['vocab'] = vocab
        kwargs['embedding'] = embedding
        self.__dict__.update(kwargs)

    def add_spec(self, **kwargs):
        """Adds new field(s).
        """
        self.__dict__.update(kwargs)

    def get_ith_data_spec(self, i):
        """Returns an instance of :class:`_DataSpec` that contains the
        `i`-th specifications.
        """
        kwargs = {}
        for k, v in six.iteritems(self.__dict__):
            kwargs[k] = v[i] if isinstance(v, (tuple, list)) else v
        return _DataSpec(**kwargs)

    def set_ith_data_spec(self, i, data_spec, total_count):
        """Sets the `i`-th specification to respective values in
        :attr:`data_spec`.
        """
        for k, v in six.iteritems(data_spec.__dict__):
            if k in self.__dict__:
                v_ = self.__dict__[k]
                if isinstance(v_, (tuple, list)):
                    v_[i] = v
                else:
                    new_v_ = [v_] * total_count
                    new_v_[i] = v
                    self.__dict__[k] = new_v_
            else:
                v_ = [None] * total_count
                v_[i] = v
                self.__dict__[k] = v_

def _make_length_filter_fn(length_name, max_length):
    """Returns a predicate function which takes in data sample
    and returns a bool indicating whether to filter by length.
    """
    def _filter_fn(data):
        return data[length_name] <= max_length
    return _filter_fn

def _make_smaller_batch_filter_fn(batch_size):
    """Returns a predicate function which takes in a batched data
    and returns a bool indicating whether the batch is of :attr:`batch_size`.
    """
    def _filter_fn(data):
        if isinstance(data, (list, tuple)):
            return _filter_fn(data[0])
        elif isinstance(data, dict):
            return _filter_fn(data[next(iter(data))])
        else:
            return tf.equal(tf.shape(data)[0], batch_size)

    return _filter_fn

def _make_combined_filter_fn(filter_fns, mode="and"):
    """Returns a new predicate function that combines multiple
    predicate functions with certain mode.

    Returns `None` if all elements in :attr:`filter_fns` are `None`.

    Args:
        filter_fns (list): Filter functions to combine. `None` functions are
            ignored.
        mode (str): A mode from `{"and", "or"}`.
    """
    if not any(filter_fns):
        return None

    def _combined_fn(data):
        outputs = []
        for fn in filter_fns:
            if fn:
                outputs.append(fn(data))
        if mode == "and":
            return tf.reduce_all(outputs)
        elif mode == "or":
            return tf.reduce_any(outputs)
        else:
            raise ValueError("Unknown mode: {}".format(mode))
    return _combined_fn

def _connect_name(lhs_name, rhs_name):
    if not lhs_name:
        return rhs_name
    if not rhs_name:
        return lhs_name
    return "{}_{}".format(lhs_name, rhs_name)

def maybe_tuple(data):
    """Returns `tuple(data)` if :attr:`data` contains more than 1 elements.

    Used to wrap `map_func` inputs.
    """
    data = tuple(data)
    data = data if len(data) > 1 else data[0]
    return data

def make_partial(fn, *args, **kwargs):
    """Returns a new function with single argument by freezing other arguments
    of :attr:`fn`.
    """
    def _new_fn(data):
        return fn(data, *args, **kwargs)
    return _new_fn

def name_prefix_fn(name_prefix):
    """Returns a function that append a prefix to field names.
    """
    def _prefix_fn(data):
        transformed_data = {}
        for name, value in six.iteritems(data):
            new_name = _connect_name(name_prefix, name)
            transformed_data[new_name] = value
        return transformed_data

    return _prefix_fn

def make_chained_transformation(tran_fns, *args, **kwargs):
    """Returns a dataset transformation function that applies a list of
    transformations sequentially.

    Args:
        tran_fns (list): A list of dataset transformation function.
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

def make_combined_transformation(tran_fns, name_prefix=None, *args, **kwargs):
    """Returns a dataset transformation function that applies
    transformations to each component of the data.

    The data to be transformed must be a tuple of the same length
    of :attr:`tran_fns`.

    Args:
        tran_fns (list): A list of elements where each element is a
            transformation function or a list of transformation functions.
        name_prefix (list, optional): Prefix to the field names of each
            component of the data, to prevent fields with the same name
            in different components from overriding each other. If not `None`,
            must be of the same length of :attr:`tran_fns`.
        *args: Extra arguments for each of the transformation function.
        **kwargs: Extra keyword arguments for each of the transformation
            function.

    Returns:
        A transformation function to be used in
        :tf_main:`tf.data.Dataset.map <data/Dataset#map>`.
    """
    if name_prefix and len(name_prefix) != len(tran_fns):
        raise ValueError("`name_prefix`, if provided, must be of the same "
                         "length of `tran_fns`.")

    def _combined_fn(data):
        transformed_data = {}
        for i, tran_fns_i in enumerate(tran_fns):
            data_i = data[i]
            # Process data_i
            if not isinstance(tran_fns_i, (list, tuple)):
                tran_fns_i = [tran_fns_i]
            for tran_fns_ij in tran_fns_i:
                data_i = tran_fns_ij(data_i, *args, **kwargs)
            # Add to dict by appending name prefix
            for name, value in six.iteritems(data_i):
                new_name = name
                if name_prefix:
                    new_name = _connect_name(name_prefix[i], name)
                if new_name in transformed_data:
                    raise ValueError(
                        "Field name already exists: {}".format(new_name))
                transformed_data[new_name] = value
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

