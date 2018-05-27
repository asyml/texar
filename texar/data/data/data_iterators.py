#
"""
Various data iterator classes.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf

import texar as tx

__all__ = [
    "DataIterator",
    "TrainTestDataIterator"
]

class DataIterator(object):
    """Data iterator that iterates through multiple datasets.

    Args:
        datasets: Datasets to iterates through. This can be:

        - A single instance of :tf_main:`tf.data.Dataset <data/Dataset>` or \
          :class:`~texar.data.DataBase`.
        - A `dict` that maps dataset name to \
          instance of :tf_main:`tf.data.Dataset <data/Dataset>` or \
          :class:`~texar.data.DataBase`.
        - A list of texar Data instances that inherit \
          :class:`texar.data.DataBase`. The name of each \
          of the instances must be unique.
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

        arb_dataset = self._datasets[next(iter(self._datasets))]
        self._iterator = tf.data.Iterator.from_structure(
            arb_dataset.output_types, arb_dataset.output_shapes)
        self._iterator_init_ops = {
            name: self._iterator.make_initializer(d)
            for name, d in self._datasets.items()}

    def switch_to_dataset(self, sess, dataset_name=None):
        """Re-initializes the iterator from a given dataset to start iterating
        over the dataset.

        Args:
            sess: The current tf session.
            dataset_name (optional): Name of the dataset. If not provided,
                there must be only one Dataset.
        """
        if dataset_name is None:
            if self.num_datasets > 1:
                raise ValueError("`dataset_name` is required if there are "
                                 "more than one datasets.")
            dataset_name = self._default_dataset_name
        if dataset_name not in self._datasets:
            raise ValueError("Dataset not found: ", dataset_name)
        sess.run(self._iterator_init_ops[dataset_name])

    def get_next(self):
        """Returns the next element of the activated dataset.
        """
        return self._iterator.get_next()

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

class TrainTestDataIterator(DataIterator):
    """Data iterator that alternatives between train, val, and test datasets.

    :attr:`train`, :attr:`val`, and :attr:`test` can be instance of
    :tf_main:`tf.data.Dataset <data/Dataset>` or :class:`~texar.data.DataBase`.
    At least one of them must be provided.

    Args:
        train (optional): Training data.
        val (optional): Validation data.
        test (optional): Test data.
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
        """Starts to iterate through training data

        Args:
            sess: The current tf session.
        """
        if self._train_name not in self._datasets:
            raise ValueError("Training data not provided.")
        self.switch_to_dataset(sess, self._train_name)

    def switch_to_val_data(self, sess):
        """Starts to iterate through val data

        Args:
            sess: The current tf session.
        """
        if self._val_name not in self._datasets:
            raise ValueError("Val data not provided.")
        self.switch_to_dataset(sess, self._val_name)

    def switch_to_test_data(self, sess):
        """Starts to iterate through test data

        Args:
            sess: The current tf session.
        """
        if self._test_name not in self._datasets:
            raise ValueError("Test data not provided.")
        self.switch_to_dataset(sess, self._test_name)



