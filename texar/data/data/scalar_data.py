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

from texar.data.data_utils import count_file_lines
from texar.data.data import dataset_utils as dsutils
from texar.data.data.data_base import DataBase
from texar.data.data.mono_text_data import MonoTextData
from texar.data.data_decoders import ScalarDataDecoder

# pylint: disable=invalid-name, arguments-differ, not-context-manager

__all__ = [
    "_default_scalar_dataset_hparams",
    "ScalarData"
]

def _default_scalar_dataset_hparams():
    """Returns hyperparameters of a scalar dataset with default values.

    See :meth:`texar.data.ScalarData.default_hparams` for details.
    """
    return {
        "files": [],
        "compression_type": None,
        "data_type": "int",
        "data_name": None,
        "other_transformations": [],
        "@no_typecheck": ["files"]
    }

class ScalarData(DataBase):
    """Scalar data where each line of the files is a scalar (int or float),
    e.g., a data label.

    Args:
        hparams (dict): Hyperparameters. See :meth:`default_hparams` for the
            defaults.

    The processor reads and processes raw data and results in a TF dataset
    whose element is a python `dict` including one field. The field name is
    specified in :attr:`hparams["dataset"]["data_name"]`. If not specified,
    the default name is `"data"`. The field name can be accessed through
    :attr:`data_name`.

    This field is a Tensor of shape `[batch_size]` containing a batch of
    scalars, of either int or float type as specified in :attr:`hparams`.

    Example:

        .. code-block:: python

            hparams={
                'dataset': { 'files': 'data.txt', 'data_name': 'label' },
                'batch_size': 2
            }
            data = ScalarData(hparams)
            iterator = DataIterator(data)
            batch = iterator.get_next()

            iterator.switch_to_dataset(sess) # initializes the dataset
            batch_ = sess.run(batch)
            # batch_ == {
            #     'label': [2, 9]
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
                # (1) Hyperparams specific to scalar dataset
                "dataset": {
                    "files": [],
                    "compression_type": None,
                    "data_type": "int",
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
                "name": "scalar_data",
            }

        Here:

        1. For the hyperparameters in the :attr:`"dataset"` field:

            "files" : str or list
                A (list of) file path(s).

                Each line contains a single scalar number.

            "compression_type" : str, optional
                One of "" (no compression), "ZLIB", or "GZIP".

            "data_type" : str
                The scalar type. Currently supports "int" and "float".

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
        hparams["name"] = "scalar_data"
        hparams.update({
            "dataset": _default_scalar_dataset_hparams()
        })
        return hparams

    @staticmethod
    def _get_dtype(dtype_hparam):
        if dtype_hparam == "int":
            dtype = tf.int32
        elif dtype_hparam == "float":
            dtype = tf.float32
        else:
            raise ValueError("Unknown data type: " + dtype_hparam)
        return dtype

    @staticmethod
    def _make_processor(dataset_hparams, data_spec, chained=True,
                        name_prefix=None):
        # Create data decoder
        decoder = ScalarDataDecoder(
            ScalarData._get_dtype(dataset_hparams["data_type"]),
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
        dataset_hparams = self._hparams.dataset

        # Create and shuffle dataset
        dataset = MonoTextData._make_mono_text_dataset(dataset_hparams)
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

    def list_items(self):
        """Returns the list of item names that the data can produce.

        Returns:
            A list of strings.
        """
        return list(self._dataset.output_types.keys())

    @property
    def dataset(self):
        """The dataset.
        """
        return self._dataset

    def dataset_size(self):
        """Returns the number of data instances in the dataset.

        Note that this is the total data count in the raw files, before any
        filtering and truncation.
        """
        if not self._dataset_size:
            # pylint: disable=attribute-defined-outside-init
            self._dataset_size = count_file_lines(
                self._hparams.dataset.files)
        return self._dataset_size

    @property
    def data_name(self):
        """The name of the data tensor, "data" by default if not specified in
        :attr:`hparams`.
        """
        return self._decoder.data_tensor_name

