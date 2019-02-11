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
Various data classes that define data reading, parsing, batching, and other
preprocessing operations.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf

from texar.data.data import dataset_utils as dsutils
from texar.data.data.data_base import DataBase
from texar.data.data.mono_text_data import MonoTextData
from texar.data.data_decoders import TFRecordDataDecoder

# pylint: disable=invalid-name, arguments-differ, not-context-manager

__all__ = [
    "_default_tfrecord_dataset_hparams",
    "TFRecordData"
]

def _default_tfrecord_dataset_hparams():
    """Returns hyperparameters of a TFRecord dataset with default values.

    See :meth:`texar.data.TFRecordData.default_hparams` for details.
    """
    """
    Add new features like `operations_and_keys`
    TODO
    """
    return {
        "files": [],
        "feature_key_and_types": {},
        "operations_and_keys": {},
        "compression_type": None,
        "data_name": None,
        "other_transformations": [],
        "@no_typecheck": ["files", "feature_key_and_types"],
    }

class TFRecordData(DataBase):
    """TFRecord data where each line of the files is a single data batch content,
    e.g., an image example.

    Args:
        hparams (dict): Hyperparameters. See :meth:`default_hparams` for the
            defaults.

    The processor reads and process raw data in a TF dataset, the TFRecords file,
    whose element can be the data for the traning or testing data feeds. 
    Along with the TF dataset the processor also construct the feature description to 
    parse TFRecord data through reading the feature, which is specified in 
    :attr:`hparams["dataset"]["feature_key_and_types"]`. User should input `dict` with
    key as the feature name in `str`, and value as the feature Dtype in `str`, the feature 
    names and Dtype pairs should match the TFRecords file.
    

    Example:

        .. code-block:: python

            hparams={
                'dataset': { 
                    'files': 'image.tfrecord', 
                    'feature_key_and_types': {
                        'height': 'tf.int64',
                        'width': 'tf.int64',
                        'label': 'tf.int64',
                        'image_raw': 'tf.string'
                    }
                },
                'batch_size': 1
            }
            data = TFRecordData(hparams)
            iterator = DataIterator(data)
            batch = iterator.get_next()

            iterator.switch_to_dataset(sess) # initializes the dataset
            batch_ = sess.run(batch)
            # batch_ == {
            #    'data': {
            #        'height': [239],
            #        'width': [149],
            #        'label': [1],
            #        'image_raw': ['...'],
            #    }
            # }
    """

    def __init__(self, hparams):
        DataBase.__init__(self, hparams)
        with tf.name_scope(self.name, self.default_hparams()["name"]):
            self._make_data()

    def _read_TFRecord_data(self):
        filenames = self._hparams.dataset.files
        dataset = tf.data.TFRecordDataset(filenames=filenames)
        return dataset
    @staticmethod
    def _make_processor(dataset_hparams, data_spec, chained=True,
                        name_prefix=None):
        # Create data decoder
        decoder = TFRecordDataDecoder(
            feature_key_and_dtype=dataset_hparams.feature_key_and_types,
            data_name=name_prefix)
        # Create other transformations
        data_spec.add_spec(decoder=decoder)
        # pylint: disable=protected-access
        other_trans = MonoTextData._make_other_transformations(
            dataset_hparams["other_transformations"], data_spec)

        data_spec.add_spec(name_prefix=name_prefix)

        if chained:
            chained_tran = dsutils.make_chained_transformation(
                [decoder] + other_trans)
            return chained_tran, data_spec
        else:
            return decoder, other_trans, data_spec

    def _process_dataset(self, dataset, hparams, data_spec):
        chained_tran, data_spec = self._make_processor(
            hparams["dataset"], data_spec,
            name_prefix=hparams["dataset"]["data_name"])
        num_parallel_calls = hparams["num_parallel_calls"]
        dataset = dataset.map(
            lambda *args: chained_tran(dsutils.maybe_tuple(args)),
            num_parallel_calls=num_parallel_calls)

        # Truncates data count
        dataset = dataset.take(hparams["max_dataset_size"])

        return dataset, data_spec

    def _make_data(self):
        dataset = self._read_TFRecord_data()
        # Create and shuffle dataset
        dataset, dataset_size = self._shuffle_dataset(
            dataset, self._hparams, self._hparams.dataset.files)
        self._dataset_size = dataset_size

        # Processing
        # pylint: disable=protected-access
        data_spec = dsutils._DataSpec(dataset=dataset,
                                      dataset_size=self._dataset_size)
        dataset, data_spec = self._process_dataset(dataset, self._hparams,
                                                   data_spec)
        self._data_spec = data_spec
        self._decoder = data_spec.decoder # pylint: disable=no-member

        # Batching
        dataset = self._make_batch(dataset, self._hparams)

        # Prefetching
        if self._hparams.prefetch_buffer_size > 0:
            dataset = dataset.prefetch(self._hparams.prefetch_buffer_size)

        self._dataset = dataset
        self.dataset = dataset

    def list_items(self):
        """Returns the list of item names that the data can produce.

        Returns:
            A list of strings.
        """
        return list(self._dataset.output_types.keys())

    @staticmethod
    def default_hparams():
        """Returns a dicitionary of default hyperparameters.

        .. code-block:: python

            {
                # (1) Hyperparams specific to scalar dataset
                'dataset': { 
                    'files': [], 
                    'feature_key_and_types': {},
                    "compression_type": None,
                    "data_name": None,
                    "other_transformations": [],
                }
                # (2) General hyperparams
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
                "name": "tfrecord_data",
            }

        Here:

        1. For the hyperparameters in the :attr:`"dataset"` field:

            "files" : str or list
                A (list of) file path(s).

                Path to the TFRecords file, each line contains a single data batch.

            "feature_key_and_types" : dict
                The feature name keys and their data types, both keys and types are in `str`

            "compression_type" : str, optional
                One of "" (no compression), "ZLIB", or "GZIP".

            "other_transformations" : list
                A list of transformation functions or function names/paths to
                further transform each single data instance.

                (More documentations to be added.)

            "data_name" : str
                Name of the dataset.

        2. For the **general** hyperparameters, see
        :meth:`texar.data.DataBase.default_hparams` for details.

        """
        hparams = DataBase.default_hparams()
        hparams["name"] = "tfrecord_data"
        hparams.update({
            "dataset": _default_tfrecord_dataset_hparams()
        })
        return hparams
