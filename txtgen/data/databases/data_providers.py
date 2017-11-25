#
"""
:class:`~tensorflow.contrib.slim.python.slim.data.data_provider.DataProvider`
instances that provide a single example of requested data.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.training import queue_runner  # pylint: disable=E0611
import tensorflow.contrib.slim as tf_slim
from tensorflow.contrib.slim.python.slim.data import parallel_reader

# pylint: disable=too-many-arguments, too-many-locals

__all__ = [
    "ParallelDataProvider"
]

class ParallelDataProvider(tf_slim.data_provider.DataProvider):
    """A DataProvider that reads multiple aligned datasets.

    Args:
        datasets: A list of :class:`Dataset` instances. The provider reads
            one element from each of the datasets every time.
        reader_kwargs (optional): A list of dictionaries or `None`. Each
            dictionary contains keyword arguments for the reader of respective
            dataset in :attr:`datatsets`. If not `None`,
            :attr:`reader_kwargs` must have the same length with
            :attr:`datasets`.
        dtypes (list, optional): Types of the data in each of the datasets.
            If `None` (default), types of all datasets are assumed to be
            `tf.string`. If not `None`, :attr:`dtypes` must have the same length
            with :attr:`datasets`.
        shuffle (bool): Whether to shuffle the data sources and common queue
            when reading.
        num_epochs (int, optional): The number of times each data source is
            read. If `None` (default), the data will be cycled through
            indefinitely.
        common_queue_capacity (int): The capacity of the common queue.
        common_queue_min (int): The minimum number of elements in the
            common queue after a dequeue. Needed for a good shuffle.
        seed (int, optional): The seed to use if shuffling.
        scope (str, optional): Optional name scope for the ops.
    """

    def __init__(self,
                 datasets,
                 reader_kwargs=None,
                 dtypes=None,
                 shuffle=True,
                 num_epochs=None,
                 common_queue_capacity=1024,
                 common_queue_min=526,
                 seed=None,
                 scope=None):
        scope = scope or "parallel_data_provider"

        if not isinstance(datasets, list) or len(datasets) < 2:
            raise ValueError("`datasets` must be a list of length >= 2.")

        if reader_kwargs is None:
            reader_kwargs = [None for _ in range(len(datasets))]
        elif not isinstance(reader_kwargs, list) or \
                len(reader_kwargs) != len(datasets):
            raise ValueError(
                "If `reader_kwargs` is not `None`, it must be a list of the "
                "same length with `datasets`.")

        if dtypes is None:
            dtypes = [tf.string for _ in range(len(datasets))]
        elif not isinstance(dtypes, list) or len(dtypes) != len(datasets):
            raise ValueError(
                "If `dtypes` is not `None`, it must be a list of the "
                "same length with `datasets`.")

        data_list = []
        for dataset, reader_kwargs in zip(datasets, reader_kwargs):
            _, data = parallel_reader.parallel_read(
                dataset.data_sources,
                reader_class=dataset.reader,
                num_epochs=num_epochs,
                num_readers=1,
                # Use one reader to ensure aligned source-target data
                reader_kwargs=reader_kwargs,
                shuffle=False,
                capacity=common_queue_capacity,
                min_after_dequeue=common_queue_min,
                scope=scope)
            data_list.append(data)

        if shuffle:
            with tf.name_scope(scope):  # pylint: disable=not-context-manager
                random_shuffle_queue = tf.RandomShuffleQueue(
                    capacity=common_queue_capacity,
                    min_after_dequeue=common_queue_min,
                    dtypes=dtypes,
                    seed=seed,
                    name="shuffle_queue")
                enqueue_ops = [random_shuffle_queue.enqueue(data_list)]
                queue_runner.add_queue_runner(
                    queue_runner.QueueRunner(random_shuffle_queue, enqueue_ops))
                data_list = random_shuffle_queue.dequeue()

        items_list = []
        tensors_list = []
        for dataset, data in zip(datasets, data_list):
            items = dataset.decoder.list_items()
            tensors = dataset.decoder.decode(data, items)
            items_list += items
            tensors_list += tensors

        super(ParallelDataProvider, self).__init__(
            items_to_tensors=dict(zip(items_list, tensors_list)),
            num_samples=datasets[0].num_samples)
