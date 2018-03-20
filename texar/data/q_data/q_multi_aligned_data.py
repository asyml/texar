#
"""
Various database classes that define data reading, parsing, batching, and other
preprocessing operations.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
import tensorflow.contrib.slim as tf_slim

from texar.hyperparams import HParams
from texar.core import utils
from texar.data.q_data.q_data_base import qDataBase
from texar.data.q_data import q_mono_text_data
from texar.data.q_data.q_data_providers import ParallelDataProvider
from texar.data import data_decoders
from texar.data.vocabulary import Vocab
from texar.data.embedding import Embedding
from texar.data import constants

# pylint: disable=invalid-name, arguments-differ, not-context-manager
# pylint: disable=protected-access, no-member, too-many-locals
# pylint: disable=too-many-branches

__all__ = [
    "_default_q_dataset_hparams",
    "qMultiAlignedData"
]

def _default_q_dataset_hparams():
    """Returns hyperparameters of a dataset with default values.
    """
    # TODO(zhiting): add more docs
    hparams = q_mono_text_data._default_q_mono_text_dataset_hparams()
    hparams.update({
        "data_type": "text",
        "data_name_prefix": None,
        "vocab_share_with": None,
        "embedding_init_share_with": None,
        "reader_share_with": None,
        "processing_share_with": None,
    })
    return hparams

class qMultiAlignedData(qDataBase):
    """Data consists of multiple aligned parts.

    Args:
        hparams (dict): Hyperparameters. See :meth:`default_hparams` for the
            defaults.
    """

    def __init__(self, hparams):
        qDataBase.__init__(self, hparams)
        # Defaultizes hparams of each dataset
        datasets_hparams = self._hparams.datasets
        defaultized_datasets_hparams = []
        for ds_hp in datasets_hparams:
            defaultized_ds_hp = HParams(ds_hp, _default_q_dataset_hparams())
            defaultized_datasets_hparams.append(defaultized_ds_hp)
        self._hparams.datasets = defaultized_datasets_hparams

        with tf.name_scope(self.name, self.default_hparams()["name"]):
            self._datasets = self.make_dataset(self._hparams.datasets)
            self._data_provider = self._make_data_provider(self._datasets)

    @staticmethod
    def default_hparams():
        """Returns a dicitionary of default hyperparameters.
        """
        hparams = qDataBase.default_hparams()
        hparams["name"] = "multi_aligned_data"
        hparams["datasets"] = [_default_q_dataset_hparams()]
        return hparams

    @staticmethod
    def _make_text_dataset(hparams, datasets_hparams, datasets):
        dataset_id = len(datasets)

        # Get processing hparams
        proc_hparams = hparams["processing"]
        proc_share_id = hparams["processing_share_with"]
        if proc_share_id is not None:
            if proc_share_id >= dataset_id:
                raise ValueError(
                    "Cannot share processing of dataset %d with dataset %d."
                    "Must share only with a preceding dataset."
                    % (dataset_id, proc_share_id))
            proc_hparams = datasets_hparams[proc_share_id]["processing"]

        # Make vocabulary
        bos_token = utils.default_string(proc_hparams["bos_token"],
                                         constants.BOS_TOKEN)
        eos_token = utils.default_string(proc_hparams["eos_token"],
                                         constants.EOS_TOKEN)
        vocab_share_id = hparams["vocab_share_with"]
        if vocab_share_id is not None:
            if vocab_share_id >= dataset_id:
                raise ValueError(
                    "Cannot share vocab of dataset %d with dataset %d."
                    "Must share only with a preceding dataset."
                    % (dataset_id, vocab_share_id))
            ref_dataset = datasets[vocab_share_id]
            if not hasattr(ref_dataset, 'vocab'):
                raise ValueError(
                    "Cannot share vocab of dataset %d with dataset %d."
                    "Dataset %d does not have a vocab."
                    % (dataset_id, vocab_share_id, vocab_share_id))
            if bos_token == ref_dataset.vocab.bos_token and \
                    eos_token == ref_dataset.vocab.eos_token:
                vocab = ref_dataset.vocab
            else:
                vocab = Vocab(datasets_hparams[vocab_share_id]["vocab_file"],
                              bos_token=bos_token,
                              eos_token=eos_token)
        else:
            vocab = Vocab(hparams["vocab_file"],
                          bos_token=bos_token,
                          eos_token=eos_token)

        # Get reader class
        reader_share_id = hparams["reader_share_with"]
        if reader_share_id is not None:
            if reader_share_id >= dataset_id:
                raise ValueError(
                    "Cannot share reader of dataset %d with dataset %d."
                    "Must share only with a preceding dataset."
                    % (dataset_id, reader_share_id))
            reader_class = utils.get_class(
                datasets_hparams[reader_share_id]["reader"]["type"],
                ["tensorflow"])
        else:
            reader_class = utils.get_class(
                hparams["reader"]["type"], ["tensorflow"])

        # Get data decoder
        data_name_prefix = "%d" % dataset_id
        if hparams["data_name_prefix"]:
            data_name_prefix += "_" + hparams["data_name_prefix"]

        decoder = data_decoders.TextDataDecoder(
            split_level=proc_hparams["split_level"],
            delimiter=proc_hparams["delimiter"],
            bos_token=proc_hparams["bos_token"],
            eos_token=proc_hparams["eos_token"],
            max_seq_length=proc_hparams["max_seq_length"],
            token_to_id_map=vocab.token_to_id_map,
            text_tensor_name=data_name_prefix + '_text',
            length_tensor_name=data_name_prefix + '_length',
            text_id_tensor_name=data_name_prefix + '_text_ids')

        # Load embedding (optional)
        emb_share_id = hparams["embedding_init_share_with"]
        if emb_share_id is not None:
            if emb_share_id >= dataset_id:
                raise ValueError(
                    "Cannot share embedding_init of dataset %d with dataset %d."
                    "Must share only with a preceding dataset."
                    % (dataset_id, emb_share_id))
            ref_dataset = datasets[emb_share_id]
            if not hasattr(ref_dataset, 'embedding'):
                raise ValueError(
                    "Cannot share embedding_init of dataset %d with dataset %d."
                    "Dataset %d does not have an embedding."
                    % (dataset_id, emb_share_id, emb_share_id))
            embedding = ref_dataset.embedding
        else:
            emb_hparams = hparams["embedding_init"]
            embedding = None
            if emb_hparams["file"] is not None and len(emb_hparams["file"]) > 0:
                embedding = Embedding(
                    vocab.token_to_id_map_py, emb_hparams.todict())

        # Create the dataset
        dataset = tf_slim.dataset.Dataset(
            data_sources=hparams["files"],
            reader=reader_class,
            decoder=decoder,
            num_samples=None,
            items_to_descriptions=None,
            vocab=vocab,
            embedding=embedding)

        return dataset

    @staticmethod
    def _make_scalar_dataset(hparams, datasets_hparams, datasets):
        dataset_id = len(datasets)

        # Get reader class
        reader_share_id = hparams["reader_share_with"]
        if reader_share_id is not None:
            if reader_share_id >= dataset_id:
                raise ValueError(
                    "Cannot share reader of dataset %d with dataset %d."
                    "Must share only with a preceding dataset."
                    % (dataset_id, reader_share_id))
            reader_class = utils.get_class(
                datasets_hparams[reader_share_id]["reader"]["type"],
                ["tensorflow"])
        else:
            reader_class = utils.get_class(
                hparams["reader"]["type"], ["tensorflow"])

        # Get decoder
        data_name_prefix = "%d" % dataset_id
        if hparams["data_name_prefix"]:
            data_name_prefix += "_" + hparams["data_name_prefix"]

        if hparams["data_type"] == "int":
            dtype = tf.int32
        elif hparams["data_type"] == "float":
            dtype = tf.float32
        else:
            raise ValueError("Unknown data type: " + hparams["data_type"])

        decoder = data_decoders.ScalarDataDecoder(dtype, data_name_prefix)

        # Create the dataset
        dataset = tf_slim.dataset.Dataset(
            data_sources=hparams["files"],
            reader=reader_class,
            decoder=decoder,
            num_samples=None,
            items_to_descriptions=None)

        return dataset

    @staticmethod
    def make_dataset(datasets_hparams):
        datasets = []
        for ds_hp in datasets_hparams:
            if ds_hp["data_type"] == "text":
                dataset = qMultiAlignedData._make_text_dataset(
                    ds_hp, datasets_hparams, datasets)
            elif ds_hp["data_type"] == "int" or ds_hp["data_type"] == "float":
                dataset = qMultiAlignedData._make_scalar_dataset(
                    ds_hp, datasets_hparams, datasets)
            else:
                raise ValueError("Unknown data type: " + ds_hp["data_type"])

            datasets.append(dataset)

        return datasets

    def _make_data_provider(self, datasets):
        reader_kwargs_list = []
        for ds_id in range(len(datasets)):
            reader_kwargs = self._hparams.datasets[ds_id]["reader"]["kwargs"]
            if len(reader_kwargs) > 0:
                reader_kwargs = reader_kwargs.todict()
            else:
                reader_kwargs = None
            reader_kwargs_list.append(reader_kwargs)

        data_provider = ParallelDataProvider(
            datasets=datasets,
            reader_kwargs=reader_kwargs_list,
            shuffle=self._hparams.shuffle,
            num_epochs=self._hparams.num_epochs,
            common_queue_capacity=1024,
            common_queue_min=526,
            seed=self._hparams.seed)

        return data_provider

    def __call__(self):
        data = self._data_provider.get(self.list_items())
        data = dict(zip(self.list_items(), data))

        num_threads = 1
        # Recommended capacity =
        # (num_threads + a small safety margin) * batch_size + margin
        capacity = (num_threads + 32) * self._hparams.batch_size + 1024

        allow_smaller_final_batch = self._hparams.allow_smaller_final_batch

        if len(self._hparams.bucket_boundaries) == 0:
            data_batch = tf.train.batch(
                tensors=data,
                batch_size=self._hparams.batch_size,
                num_threads=num_threads,
                capacity=capacity,
                enqueue_many=False,
                dynamic_pad=True,
                allow_smaller_final_batch=allow_smaller_final_batch,
                name="%s/batch" % self.name)
        else:
            input_length = data[self._src_dataset.decoder.length_tensor_name]
            _, data_batch = tf.contrib.training.bucket_by_sequence_length(
                input_length=input_length,
                tensors=data,
                batch_size=self._hparams.batch_size,
                bucket_boundaries=self._hparams.bucket_boundaries,
                num_threads=num_threads,
                capacity=capacity,
                dynamic_pad=True,
                allow_smaller_final_batch=allow_smaller_final_batch,
                keep_input=input_length > 0,
                name="%s/bucket_batch" % self.name)

        return data_batch

    def list_items(self):
        """Returns the list of item names that the database can produce.

        Returns:
            A list of strings.
        """
        return list(self._data_provider.list_items())

    @property
    def datasets(self):
        """A list of datasets.
        """
        return self._datasets
