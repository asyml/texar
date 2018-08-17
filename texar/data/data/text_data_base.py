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
Base text data class that is enherited by all text data classes.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf

from texar.data.data.data_base import DataBase
from texar.data.data import dataset_utils as dsutils

# pylint: disable=protected-access, arguments-differ

__all__ = [
    "TextDataBase"
]

class TextDataBase(DataBase): # pylint: disable=too-few-public-methods
    """Base class inheritted by all text data classes.
    """

    def __init__(self, hparams):
        DataBase.__init__(self, hparams)

    @staticmethod
    def default_hparams():
        """Returns a dictionary of default hyperparameters.

        See the specific subclasses for the details.
        """
        hparams = DataBase.default_hparams()
        hparams.update({
            "bucket_boundaries": [],
            "bucket_batch_sizes": None,
            "bucket_length_fn": None})
        return hparams

    @staticmethod
    def _make_batch(dataset, hparams, element_length_func,
                    padded_shapes=None, padding_values=None):
        dataset = dataset.repeat(hparams.num_epochs)

        batch_size = hparams["batch_size"]
        bucket_boundaries = hparams["bucket_boundaries"]
        if padded_shapes is None:
            padded_shapes = dataset.output_shapes

        if len(bucket_boundaries) == 0:
            if hparams["allow_smaller_final_batch"]:
                dataset = dataset.padded_batch(
                    batch_size, padded_shapes, padding_values=padding_values)
            else:
                dataset = dataset.apply(
                    tf.contrib.data.padded_batch_and_drop_remainder(
                        batch_size, padded_shapes,
                        padding_values=padding_values))
        else:
            bucket_batch_size = hparams["bucket_batch_sizes"]
            if bucket_batch_size is None:
                bucket_batch_size = [batch_size] * (len(bucket_boundaries) + 1)
            dataset = dataset.apply(tf.contrib.data.bucket_by_sequence_length(
                element_length_func, bucket_boundaries, bucket_batch_size,
                padded_shapes=padded_shapes, padding_values=padding_values))
            if not hparams["allow_smaller_final_batch"]:
                if len(set(bucket_batch_size)) > 1:
                    raise ValueError(
                        "Batch size of every bucket must be the same if "
                        "smaller final batch is not allowed.")
                batch_size = bucket_batch_size[0]
                filter_fn = dsutils._make_smaller_batch_filter_fn(batch_size)
                dataset = dataset.filter(
                    lambda *args: filter_fn(dsutils.maybe_tuple(args)))

        return dataset

