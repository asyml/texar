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
Data class that supports reading TFRecords data and data type converting.
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
    return {
        "files": [],
        "feature_original_types": {},
        "feature_convert_types": {},
        "image_options": {},
        "compression_type": None,
        "other_transformations": [],
        "num_shards": None,
        "shard_id": None,
        "data_name": None,
        "@no_typecheck": [
            "files",
            "feature_original_types",
            "feature_convert_types",
            "image_options"],
    }

class TFRecordData(DataBase):
    """TFRecord data where each line of the files is a single data batch
    content, e.g., a single image data example.

    Args:
        hparams (dict): Hyperparameters. See :meth:`default_hparams`
        for the defaults.

    The processor reads and restores data from TFRecords file and
    results in a TF Dataset whose element is a python `dict` including
    field(s) of the feature(s) data from input TFRecords files
    each field contains the restored data for the feature of the same
    name, which is from the key in
    :attr:`hparams["dataset"]["feature_original_types"]`

    Example:

        .. code-block:: python

            hparams={
                'dataset': {
                    'files': 'image.tfrecord',
                    'feature_original_types': {
                        'height': ['tf.int64', 'FixedLenFeature'],
                        'width': ['tf.int64', 'FixedLenFeature'],
                        'label': ['tf.int64', 'FixedLenFeature'],
                        'image_raw': ['tf.string', 'FixedLenFeature']
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

    @staticmethod
    def default_hparams():
        """Returns a dicitionary of default hyperparameters.

        .. code-block:: python

            {
                # (1) Hyperparams specific to TFRecord dataset
                'dataset': {
                    'files': [],
                    'feature_original_types': {},
                    'feature_convert_types': {},
                    'image_options': {},
                    "num_shards": None,
                    "shard_id": None,
                    "other_transformations": [],
                    "data_name": None,
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
                A (list of) TFRecords file path(s).

            "feature_original_types" : dict
                The feature names (str) with their data types and length types,
                key and value in pair
                `<feature_name: [dtype, feature_len_type, len]>`, type of
                `dtype` can be `tf DType <DType>` or `str`, e.g., 'tf.int32',
                'tf.float32', etc.

                - `feature_len_type` is of type `str` and can be 
                    'FixedLenFeature' or 'VarLenFeature' for fixed length 
                    features and non-fixed length features respectively.

                - `len` is optional, it is the length for the
                    'FixedLenFeature', can be a `int`.

                E.g. it can be used as follows:

                .. code-block:: python

                    ...
                    feature_original_types = {
                        "input_ids": ["tf.int64", "FixedLenFeature", 128],
                        "input_mask": ["tf.int64", "FixedLenFeature", 128],
                        "segment_ids": ["tf.int64", "FixedLenFeature", 128],
                        "label_ids": ["tf.int64", "FixedLenFeature"]
                    }
                    ...

            "feature_convert_types" : dict, optional
                The feature names (str) with data types they are converted into,
                key and value in pair  `<feature_name: dtype>`, `dtype`
                can be a `tf DType <DType>` or `str`, e.g., 'tf.int32',
                'tf.float32', etc. If not set, data type conversion
                will not be performed.
                
                Be noticed that this converting process is after all the data
                are restored, `feature_original_types` has to be set firstly.

                E.g. it can be used as follows:

                .. code-block:: python

                    ...
                    feature_convert_types = {
                        "input_ids": "tf.int32",
                        "input_mask": "tf.int32",
                        "label_ids": "tf.int32",
                        "segment_ids": "tf.int32"
                    }
                    ...

            "image_options" : dict, optional
                Specifies the image feature name and performs image resizing,
                includes three fields:

                - "image_feature_name":
                    A `str`, the name of the feature which contains
                    the image data.
                - "resize_height":
                    A `int`, the height of the image after resizing.
                - "resize_width":
                    A `int`, the width of the image after resizing

                If not set, image data resizing will not be performed.
            "num_shards": int, optional
                The number of shards in distributed mode. Usually the
                shards number of the workflows.
                Combined with :attr:`"shard_id"`.
            "shard_id": int, optional
                Set the unique id to identify a shard, the module will
                process the corresponding shard of total data.
                Combined with :attr:`"num_shards"`. E.g. 2 gpus are
                distributed with equal workflows.

                For gpu 0:

                .. code-block:: python

                    ...
                    dataset: {
                        ...
                        "num_shards": 2,
                        "shard_id": 0,
                        ...
                    }
                    ...

                For gpu 1:

                .. code-block:: python

                    ...
                    dataset: {
                        ...
                        "num_shards": 2,
                        "shard_id": 1,
                        ...
                    }
                    ...

            "other_transformations" : list
                A list of transformation functions or function names/paths to
                further transform each single data instance.
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

    def _read_TFRecord_data(self):
        filenames = self._hparams.dataset.files
        dataset = tf.data.TFRecordDataset(filenames=filenames)
        return dataset

    @staticmethod
    def _make_processor(dataset_hparams, data_spec, chained=True,
                        name_prefix=None):
        # Create data decoder
        decoder = TFRecordDataDecoder(
            feature_original_types=dataset_hparams.feature_original_types,
            feature_convert_types=dataset_hparams.feature_convert_types,
            image_options=dataset_hparams.image_options)
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
        worker_num = self._hparams.dataset.num_shards
        worker_id = self._hparams.dataset.shard_id
        if worker_num is not None and worker_id is not None:
            dataset = dataset.shard(worker_num, worker_id)
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
        return sorted(list(self._dataset.output_types.keys()))
