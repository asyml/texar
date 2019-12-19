# Copyright 2019 The Texar Authors. All Rights Reserved.
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
Various data iterator classes.
"""

import tensorflow as tf

from texar.tf.data.data.data_base import DataBase


class DataIteratorBase:
    r"""Base class for all data iterator classes to inherit. A data iterator
    is a wrapper of :tf_main:`tf.data.Iterator <data/Iterator>`, and can
    switch between and iterate through **multiple** datasets.

    Args:
        datasets: Datasets to iterates through. This can be:

            - A single instance of :tf_main:`tf.data.Dataset <data/Dataset>`
              or instance of subclass of :class:`~texar.tf.data.DataBase`.
            - A `dict` that maps dataset name to
              instance of :tf_main:`tf.data.Dataset <data/Dataset>` or
              subclass of :class:`~texar.tf.data.DataBase`.
            - A `list` of instances of subclasses of
              :class:`texar.tf.data.DataBase`. The name of instances
              (:attr:`texar.tf.data.DataBase.name`) must be unique.
    """

    def __init__(self, datasets):
        self._default_dataset_name = 'data'
        if isinstance(datasets, (tf.data.Dataset, DataBase)):
            datasets = {self._default_dataset_name: datasets}
        elif isinstance(datasets, (list, tuple)):
            if any(not isinstance(d, DataBase) for d in datasets):
                raise ValueError("`datasets` must be an non-empty list of "
                                 "`tx.data.DataBase` instances.")
            num_datasets = len(datasets)
            datasets = {d.name: d for d in datasets}
            if len(datasets) < num_datasets:
                raise ValueError("Names of datasets must be unique.")

        _datasets = {}
        for k, v in datasets.items():
            _datasets[k] = v if isinstance(v, tf.data.Dataset) else v.dataset
        self._datasets = _datasets

        if len(self._datasets) <= 0:
            raise ValueError("`datasets` must not be empty.")

    @property
    def num_datasets(self):
        r"""Number of datasets.
        """
        return len(self._datasets)

    @property
    def dataset_names(self):
        r"""A list of dataset names.
        """
        return list(self._datasets.keys())


class DataIterator(DataIteratorBase):
    r"""Data iterator that switches and iterates through multiple datasets.

    This is a wrapper of TF reinitializable :tf_main:`iterator <data/Iterator>`.

    Args:
        datasets: Datasets to iterates through. This can be:

            - A single instance of :tf_main:`tf.data.Dataset <data/Dataset>`
              or instance of subclass of :class:`~texar.tf.data.DataBase`.
            - A `dict` that maps dataset name to
              instance of :tf_main:`tf.data.Dataset <data/Dataset>` or
              subclass of :class:`~texar.tf.data.DataBase`.
            - A `list` of instances of subclasses of
              :class:`texar.tf.data.DataBase`. The name of instances
              (:attr:`texar.tf.data.DataBase.name`) must be unique.

    Example:

        .. code-block:: python

            train_data = MonoTextData(hparams_train)
            test_data = MonoTextData(hparams_test)
            iterator = DataIterator({'train': train_data, 'test': test_data})
            batch = iterator.get_next()

            TODO: Should be updated.
            sess = tf.Session()

            for _ in range(200): # Run 200 epochs of train/test
                # Starts iterating through training data from the beginning
                iterator.switch_to_dataset(sess, 'train')
                while True:
                    try:
                        train_batch_ = sess.run(batch)
                    except tf.errors.OutOfRangeError:
                        print("End of training epoch.")
                # Starts iterating through test data from the beginning
                iterator.switch_to_dataset(sess, 'test')
                while True:
                    try:
                        test_batch_ = sess.run(batch)
                    except tf.errors.OutOfRangeError:
                        print("End of test epoch.")
    """

    def __init__(self, datasets):
        DataIteratorBase.__init__(self, datasets)

        first_dataset = self._datasets[sorted(self.dataset_names)[0]]
        self._iterator = tf.compat.v1.data.Iterator.from_structure(
            tf.compat.v1.data.get_output_types(first_dataset),
            tf.compat.v1.data.get_output_shapes(first_dataset))
        self._iterator_init_ops = {
            name: self._iterator.make_initializer(d)
            for name, d in self._datasets.items()
        }

    def switch_to_dataset(self, dataset_name=None):
        r"""Re-initializes the iterator of a given dataset and starts iterating
        over the dataset (from the beginning).

        Args:
            dataset_name (optional): Name of the dataset. If not provided,
                there must be only one Dataset.
        """
        if dataset_name is None:
            if self.num_datasets > 1:
                raise ValueError("`dataset_name` is required if there are "
                                 "more than one datasets.")
            dataset_name = next(iter(self._datasets))
        if dataset_name not in self._datasets:
            raise ValueError("Dataset not found: ", dataset_name)
        self._iterator.make_initializer(self._datasets[dataset_name])

    def get_next(self):
        r"""Returns the next element of the activated dataset.
        """
        return self._iterator.get_next()


class TrainTestDataIterator(DataIterator):
    r"""Data iterator that alternatives between train, val, and test datasets.

    :attr:`train`, :attr:`val`, and :attr:`test` can be instance of
    either :tf_main:`tf.data.Dataset <data/Dataset>` or subclass of
    :class:`~texar.tf.data.DataBase`. At least one of them must be provided.

    This is a wrapper of :class:`~texar.tf.data.DataIterator`.

    Args:
        train (optional): Training data.
        val (optional): Validation data.
        test (optional): Test data.

    Example:

        .. code-block:: python

            train_data = MonoTextData(hparams_train)
            val_data = MonoTextData(hparams_val)
            iterator = TrainTestDataIterator(train=train_data, val=val_data)
            batch = iterator.get_next()

            TODO: Should be updated.
            sess = tf.Session()

            for _ in range(200): # Run 200 epochs of train/val
                # Starts iterating through training data from the beginning
                iterator.switch_to_train_data(sess)
                while True:
                    try:
                        train_batch_ = sess.run(batch)
                    except tf.errors.OutOfRangeError:
                        print("End of training epoch.")
                # Starts iterating through val data from the beginning
                iterator.switch_to_val_dataset(sess)
                while True:
                    try:
                        val_batch_ = sess.run(batch)
                    except tf.errors.OutOfRangeError:
                        print("End of val epoch.")
    """

    def __init__(self, train=None, val=None, test=None):
        dataset_dict = {}
        self._train_name = 'train'
        self._val_name = 'val'
        self._test_name = 'test'
        if train is not None:
            dataset_dict[self._train_name] = train
        if val is not None:
            dataset_dict[self._val_name] = val
        if test is not None:
            dataset_dict[self._test_name] = test
        if len(dataset_dict) == 0:
            raise ValueError("At least one of `train`, `val`, and `test` "
                             "must be provided.")

        DataIterator.__init__(self, dataset_dict)

    def switch_to_train_data(self):
        r"""Starts to iterate through training data (from the beginning).
        """
        if self._train_name not in self._datasets:
            raise ValueError("Training data not provided.")
        self.switch_to_dataset(self._train_name)

    def switch_to_val_data(self):
        r"""Starts to iterate through val data (from the beginning).
        """
        if self._val_name not in self._datasets:
            raise ValueError("Val data not provided.")
        self.switch_to_dataset(self._val_name)

    def switch_to_test_data(self):
        r"""Starts to iterate through test data (from the beginning).
        """
        if self._test_name not in self._datasets:
            raise ValueError("Test data not provided.")
        self.switch_to_dataset(self._test_name)
