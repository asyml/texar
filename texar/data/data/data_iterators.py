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
Various data iterator classes.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf

import texar as tx
from texar.utils.variables import get_unique_named_variable_scope

__all__ = [
    "DataIteratorBase",
    "DataIterator",
    "TrainTestDataIterator",
    "FeedableDataIterator",
    "TrainTestFeedableDataIterator"
]

class DataIteratorBase(object):
    """Base class for all data iterator classes to inherit. A data iterator
    is a wrapper of :tf_main:`tf.data.Iterator <data/Iterator>`, and can
    switch between and iterate through **multiple** datasets.

    Args:
        datasets: Datasets to iterates through. This can be:

            - A single instance of :tf_main:`tf.data.Dataset <data/Dataset>` \
            or instance of subclass of :class:`~texar.data.DataBase`.
            - A `dict` that maps dataset name to \
            instance of :tf_main:`tf.data.Dataset <data/Dataset>` or \
            subclass of :class:`~texar.data.DataBase`.
            - A `list` of instances of subclasses of \
            :class:`texar.data.DataBase`. The name of instances \
            (:attr:`texar.data.DataBase.name`) must be unique.
    """

    def __init__(self, datasets):
        self._default_dataset_name = 'data'
        if isinstance(datasets, (tf.data.Dataset, tx.data.DataBase)):
            datasets = {self._default_dataset_name: datasets}
        elif isinstance(datasets, (list, tuple)):
            if any(not isinstance(d, tx.data.DataBase) for d in datasets):
                raise ValueError("`datasets` must be an non-empty list of "
                                 "`texar.data.DataBase` instances.")
            num_datasets = len(datasets)
            datasets = {d.name: d for d in datasets}
            if len(datasets) < num_datasets:
                raise ValueError("Names of datasets must be unique.")

        _datasets = {}
        for k, v in datasets.items(): # pylint: disable=invalid-name
            _datasets[k] = v if isinstance(v, tf.data.Dataset) else v.dataset
        self._datasets = _datasets

        if len(self._datasets) <= 0:
            raise ValueError("`datasets` must not be empty.")

    @property
    def num_datasets(self):
        """Number of datasets.
        """
        return len(self._datasets)

    @property
    def dataset_names(self):
        """A list of dataset names.
        """
        return list(self._datasets.keys())


class DataIterator(DataIteratorBase):
    """Data iterator that switches and iterates through multiple datasets.

    This is a wrapper of TF reinitializble :tf_main:`iterator <data/Iterator>`.

    Args:
        datasets: Datasets to iterates through. This can be:

            - A single instance of :tf_main:`tf.data.Dataset <data/Dataset>` \
            or instance of subclass of :class:`~texar.data.DataBase`.
            - A `dict` that maps dataset name to \
            instance of :tf_main:`tf.data.Dataset <data/Dataset>` or \
            subclass of :class:`~texar.data.DataBase`.
            - A `list` of instances of subclasses of \
            :class:`texar.data.DataBase`. The name of instances \
            (:attr:`texar.data.DataBase.name`) must be unique.

    Example:

        .. code-block:: python

            train_data = MonoTextData(hparams_train)
            test_data = MonoTextData(hparams_test)
            iterator = DataIterator({'train': train_data, 'test': test_data})
            batch = iterator.get_next()

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

        self._variable_scope = get_unique_named_variable_scope('data_iterator')
        with tf.variable_scope(self._variable_scope):
            first_dataset = self._datasets[sorted(self.dataset_names)[0]]
            self._iterator = tf.data.Iterator.from_structure(
                first_dataset.output_types, first_dataset.output_shapes)
            self._iterator_init_ops = {
                name: self._iterator.make_initializer(d)
                for name, d in self._datasets.items()
            }

    def switch_to_dataset(self, sess, dataset_name=None):
        """Re-initializes the iterator of a given dataset and starts iterating
        over the dataset (from the beginning).

        Args:
            sess: The current tf session.
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
        sess.run(self._iterator_init_ops[dataset_name])

    def get_next(self):
        """Returns the next element of the activated dataset.
        """
        return self._iterator.get_next()

class TrainTestDataIterator(DataIterator):
    """Data iterator that alternatives between train, val, and test datasets.

    :attr:`train`, :attr:`val`, and :attr:`test` can be instance of
    either :tf_main:`tf.data.Dataset <data/Dataset>` or subclass of
    :class:`~texar.data.DataBase`. At least one of them must be provided.

    This is a wrapper of :class:`~texar.data.DataIterator`.

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


    def switch_to_train_data(self, sess):
        """Starts to iterate through training data (from the beginning).

        Args:
            sess: The current tf session.
        """
        if self._train_name not in self._datasets:
            raise ValueError("Training data not provided.")
        self.switch_to_dataset(sess, self._train_name)

    def switch_to_val_data(self, sess):
        """Starts to iterate through val data (from the beginning).

        Args:
            sess: The current tf session.
        """
        if self._val_name not in self._datasets:
            raise ValueError("Val data not provided.")
        self.switch_to_dataset(sess, self._val_name)

    def switch_to_test_data(self, sess):
        """Starts to iterate through test data (from the beginning).

        Args:
            sess: The current tf session.
        """
        if self._test_name not in self._datasets:
            raise ValueError("Test data not provided.")
        self.switch_to_dataset(sess, self._test_name)

class FeedableDataIterator(DataIteratorBase):
    """Data iterator that iterates through **multiple** datasets and switches
    between datasets.

    The iterator can switch to a dataset and resume from where we
    left off last time we visited the dataset. This is a wrapper of TF
    feedable :tf_main:`iterator <data/Iterator>`.

    Args:
        datasets: Datasets to iterates through. This can be:

            - A single instance of :tf_main:`tf.data.Dataset <data/Dataset>` \
            or instance of subclass of :class:`~texar.data.DataBase`.
            - A `dict` that maps dataset name to \
            instance of :tf_main:`tf.data.Dataset <data/Dataset>` or \
            subclass of :class:`~texar.data.DataBase`.
            - A `list` of instances of subclasses of \
            :class:`texar.data.DataBase`. The name of instances \
            (:attr:`texar.data.DataBase.name`) must be unique.

    Example:

        .. code-block:: python

            train_data = MonoTextData(hparams={'num_epochs': 200, ...})
            test_data = MonoTextData(hparams_test)
            iterator = FeedableDataIterator({'train': train_data,
                                             'test': test_data})
            batch = iterator.get_next()

            sess = tf.Session()

            def _eval_epoch(): # Iterate through test data for one epoch
                # Initialize and start from beginning of test data
                iterator.initialize_dataset(sess, 'test')
                while True:
                    try:
                        fetch_dict = { # Read from test data
                            iterator.handle: Iterator.get_handle(sess, 'test')
                        }
                        test_batch_ = sess.run(batch, feed_dict=feed_dict)
                    except tf.errors.OutOfRangeError:
                        print("End of val epoch.")

            # Initialize and start from beginning of training data
            iterator.initialize_dataset(sess, 'train')
            step = 0
            while True:
                try:
                    fetch_dict = { # Read from training data
                        iterator.handle: Iterator.get_handle(sess, 'train')
                    }
                    train_batch_ = sess.run(batch, fetch_dict=fetch_dict)

                    step +=1
                    if step % 200 == 0: # Evaluate periodically
                        _eval_epoch()
                except tf.errors.OutOfRangeError:
                    print("End of training.")
    """

    def __init__(self, datasets):
        DataIteratorBase.__init__(self, datasets)

        self._variable_scope = get_unique_named_variable_scope(
            'feedable_data_iterator')
        with tf.variable_scope(self._variable_scope):
            self._handle = tf.placeholder(tf.string, shape=[], name='handle')
            first_dataset = self._datasets[sorted(self.dataset_names)[0]]
            self._iterator = tf.data.Iterator.from_string_handle(
                self._handle, first_dataset.output_types,
                first_dataset.output_shapes)

            self._dataset_iterators = {
                name: dataset.make_initializable_iterator()
                for name, dataset in self._datasets.items()
            }

    def get_handle(self, sess, dataset_name=None):
        """Returns a dataset handle used to feed the
        :attr:`handle` placeholder to fetch data from the dataset.

        Args:
            sess: The current tf session.
            dataset_name (optional): Name of the dataset. If not provided,
                there must be only one Dataset.

        Returns:
            A string handle to be fed to the :attr:`handle` placeholder.

        Example:

            .. code-block:: python

                next_element = iterator.get_next()
                train_handle = iterator.get_handle(sess, 'train')
                # Gets the next training element
                ne_ = sess.run(next_element,
                               feed_dict={iterator.handle: train_handle})
        """
        if dataset_name is None:
            if self.num_datasets > 1:
                raise ValueError("`dataset_name` is required if there are "
                                 "more than one datasets.")
            dataset_name = next(iter(self._datasets))
        if dataset_name not in self._datasets:
            raise ValueError("Dataset not found: ", dataset_name)
        return sess.run(self._dataset_iterators[dataset_name].string_handle())

    def restart_dataset(self, sess, dataset_name=None):
        """Restarts datasets so that next iteration will fetch data from
        the beginning of the datasets.

        Args:
            sess: The current tf session.
            dataset_name (optional): A dataset name or a list of dataset names
                that specifies which dataset(s) to restart. If `None`, all
                datasets are restart.
        """
        self.initialize_dataset(sess, dataset_name)

    def initialize_dataset(self, sess, dataset_name=None):
        """Initializes datasets. A dataset must be initialized before being
        used.

        Args:
            sess: The current tf session.
            dataset_name (optional): A dataset name or a list of dataset names
                that specifies which dataset(s) to initialize. If `None`, all
                datasets are initialized.
        """
        if dataset_name is None:
            dataset_name = self.dataset_names
        if not isinstance(dataset_name, (tuple, list)):
            dataset_name = [dataset_name]

        for name in dataset_name:
            sess.run(self._dataset_iterators[name].initializer)

    def get_next(self):
        """Returns the next element of the activated dataset.
        """
        return self._iterator.get_next()

    @property
    def handle(self):
        """The handle placeholder that can be fed with a dataset handle to
        fetch data from the dataset.
        """
        return self._handle

class TrainTestFeedableDataIterator(FeedableDataIterator):
    """Feedable data iterator that alternatives between train, val, and test
    datasets.

    This is a wrapper of :class:`~texar.data.FeedableDataIterator`.
    The iterator can switch to a dataset and resume from where it was
    left off when it was visited last time.

    :attr:`train`, :attr:`val`, and :attr:`test` can be instance of
    either :tf_main:`tf.data.Dataset <data/Dataset>` or subclass of
    :class:`~texar.data.DataBase`. At least one of them must be provided.

    Args:
        train (optional): Training data.
        val (optional): Validation data.
        test (optional): Test data.

    Example:

        .. code-block:: python

            train_data = MonoTextData(hparams={'num_epochs': 200, ...})
            test_data = MonoTextData(hparams_test)
            iterator = TrainTestFeedableDataIterator(train=train_data,
                                                     test=test_data)
            batch = iterator.get_next()

            sess = tf.Session()

            def _eval_epoch(): # Iterate through test data for one epoch
                # Initialize and start from beginning of test data
                iterator.initialize_test_dataset(sess)
                while True:
                    try:
                        fetch_dict = { # Read from test data
                            iterator.handle: Iterator.get_test_handle(sess)
                        }
                        test_batch_ = sess.run(batch, feed_dict=feed_dict)
                    except tf.errors.OutOfRangeError:
                        print("End of test epoch.")

            # Initialize and start from beginning of training data
            iterator.initialize_train_dataset(sess)
            step = 0
            while True:
                try:
                    fetch_dict = { # Read from training data
                        iterator.handle: Iterator.get_train_handle(sess)
                    }
                    train_batch_ = sess.run(batch, fetch_dict=fetch_dict)

                    step +=1
                    if step % 200 == 0: # Evaluate periodically
                        _eval_epoch()
                except tf.errors.OutOfRangeError:
                    print("End of training.")
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

        FeedableDataIterator.__init__(self, dataset_dict)

    def get_train_handle(self, sess):
        """Returns the handle of the training dataset. The handle can be used
        to feed the :attr:`handle` placeholder to fetch training data.

        Args:
            sess: The current tf session.

        Returns:
            A string handle to be fed to the :attr:`handle` placeholder.

        Example:

            .. code-block:: python

                next_element = iterator.get_next()
                train_handle = iterator.get_train_handle(sess)
                # Gets the next training element
                ne_ = sess.run(next_element,
                               feed_dict={iterator.handle: train_handle})
        """
        if self._train_name not in self._datasets:
            raise ValueError("Training data not provided.")
        return self.get_handle(sess, self._train_name)

    def get_val_handle(self, sess):
        """Returns the handle of the validation dataset. The handle can be used
        to feed the :attr:`handle` placeholder to fetch validation data.

        Args:
            sess: The current tf session.

        Returns:
            A string handle to be fed to the :attr:`handle` placeholder.
        """
        if self._val_name not in self._datasets:
            raise ValueError("Val data not provided.")
        return self.get_handle(sess, self._val_name)

    def get_test_handle(self, sess):
        """Returns the handle of the test dataset. The handle can be used
        to feed the :attr:`handle` placeholder to fetch test data.

        Args:
            sess: The current tf session.

        Returns:
            A string handle to be fed to the :attr:`handle` placeholder.
        """
        if self._test_name not in self._datasets:
            raise ValueError("Test data not provided.")
        return self.get_handle(sess, self._test_name)

    def restart_train_dataset(self, sess):
        """Restarts the training dataset so that next iteration will fetch
        data from the beginning of the training dataset.

        Args:
            sess: The current tf session.
        """
        if self._train_name not in self._datasets:
            raise ValueError("Training data not provided.")
        self.restart_dataset(sess, self._train_name)

    def restart_val_dataset(self, sess):
        """Restarts the validation dataset so that next iteration will fetch
        data from the beginning of the validation dataset.

        Args:
            sess: The current tf session.
        """
        if self._val_name not in self._datasets:
            raise ValueError("Val data not provided.")
        self.restart_dataset(sess, self._val_name)

    def restart_test_dataset(self, sess):
        """Restarts the test dataset so that next iteration will fetch
        data from the beginning of the test dataset.

        Args:
            sess: The current tf session.
        """
        if self._test_name not in self._datasets:
            raise ValueError("Test data not provided.")
        self.restart_dataset(sess, self._test_name)
