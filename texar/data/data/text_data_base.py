#
"""
Base text data class that is enherited by all text data classes.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf

from texar.data.data.data_base import DataBase


__all__ = [
    "TextDataBase"
]

class TextDataBase(DataBase): # pylint: disable=too-few-public-methods
    """Base class of all text data classes.
    """

    def __init__(self, hparams):
        DataBase.__init__(self, hparams)

    # TODO (zhiting): add more docs
    @staticmethod
    def default_hparams():
        """Returns a dictionary of default hyperparameters.
        """
        hparams = DataBase.default_hparams()
        hparams.update({
            "bucket_boundaries": [],
            "bucket_batch_sizes": None,
            "bucket_length_fn": None})
        return hparams

    @staticmethod
    def _make_batch(dataset, hparams, element_length_func):
        dataset = dataset.repeat(hparams.num_epochs)
        bucket_boundaries = hparams["bucket_boundaries"]
        batch_size = hparams["batch_size"]
        if len(bucket_boundaries) == 0:
            if hparams["allow_smaller_final_batch"]:
                dataset = dataset.padded_batch(
                    batch_size, dataset.output_shapes)
            else:
                dataset = dataset.apply(
                    tf.contrib.data.padded_batch_and_drop_remainder(
                        batch_size, dataset.output_shapes))
        else:
            bucket_batch_size = hparams["bucket_batch_sizes"]
            if bucket_batch_size is None:
                bucket_batch_size = [batch_size] * (len(bucket_boundaries) + 1)
            dataset = tf.contrib.data.bucket_by_sequence_length(
                element_length_func, bucket_boundaries, bucket_batch_size)
        return dataset

