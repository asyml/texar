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
class PairedDataProvider(tf_slim.data_provider.DataProvider):
    """A DataProvider that reads two aligned datasets.

    Args:
        dataset1 (Dataset): The first dataset.
        dataset2 (Dataset): The second dataset.
        reader_kwargs1 (dict, optional): Keyword args for dataset1 reader.
        reader_kwargs2 (dict, optional): Keyword args for dataset2 reader.
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
                 dataset1,
                 dataset2,
                 reader_kwargs1=None,
                 reader_kwargs2=None,
                 shuffle=True,
                 num_epochs=None,
                 common_queue_capacity=1024,
                 common_queue_min=526,
                 seed=None,
                 scope=None):
        scope = scope or "paired_data_provider"

        _, data1 = parallel_reader.parallel_read(
            dataset1.data_sources,
            reader_class=dataset1.reader,
            num_epochs=num_epochs,
            num_readers=1,
            # Use one reader to ensure aligned source-target data
            reader_kwargs=reader_kwargs1,
            shuffle=False,
            capacity=common_queue_capacity,
            min_after_dequeue=common_queue_min,
            scope=scope)

        _, data2 = parallel_reader.parallel_read(
            dataset2.data_sources,
            reader_class=dataset2.reader,
            num_epochs=num_epochs,
            num_readers=1,
            # Use one reader to ensure aligned source-target data
            reader_kwargs=reader_kwargs2,
            shuffle=False,
            capacity=common_queue_capacity,
            min_after_dequeue=common_queue_min,
            scope=scope)

        if shuffle:
            with tf.name_scope(scope):  # pylint: disable=not-context-manager
                random_shuffle_queue = tf.RandomShuffleQueue(
                    capacity=common_queue_capacity,
                    min_after_dequeue=common_queue_min,
                    dtypes=[tf.string, tf.string],
                    seed=seed,
                    name="shuffle_queue")
                enqueue_ops = [random_shuffle_queue.enqueue([data1, data2])]
                queue_runner.add_queue_runner(
                    queue_runner.QueueRunner(random_shuffle_queue, enqueue_ops))
                data1, data2 = random_shuffle_queue.dequeue()

        items1 = dataset1.decoder.list_items()
        tensors1 = dataset1.decoder.decode(data1, items1)

        items2 = dataset2.decoder.list_items()
        tensors2 = dataset2.decoder.decode(data2, items2)

        items = items1 + items2
        tensors = tensors1 + tensors2

        super(PairedDataProvider, self).__init__(
            items_to_tensors=dict(zip(items, tensors)),
            num_samples=dataset1.num_samples)
