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

from texar.core import utils
from texar.data.q_data.q_data_base import qDataBase
from texar.data.q_data import q_mono_text_data
from texar.data.q_data.q_data_providers import ParallelDataProvider
from texar.data.data_decoders import TextDataDecoder
from texar.data.vocabulary import Vocab
from texar.data.embedding import Embedding
from texar.data import constants

# pylint: disable=invalid-name, arguments-differ, not-context-manager
# pylint: disable=protected-access, no-member

__all__ = [
    "_default_q_paired_text_dataset_hparams",
    "qPairedTextData"
]

def _default_q_paired_text_dataset_hparams():
    """Returns hyperparameters of a paired text dataset with default values.
    """
    # TODO(zhiting): add more docs
    source_hparams = q_mono_text_data._default_q_mono_text_dataset_hparams()
    source_hparams["processing"]["bos_token"] = None
    target_hparams = q_mono_text_data._default_q_mono_text_dataset_hparams()
    target_hparams.update(
        {
            "vocab_share": False,
            "embedding_init_share": False,
            "reader_share": False,
            "processing_share": False
        }
    )
    return {
        "source_dataset": source_hparams,
        "target_dataset": target_hparams
    }


class qPairedTextData(qDataBase):
    """Text data base that reads source and target text.

    This is for the use of, e.g., seq2seq models.

    Args:
        hparams (dict): Hyperparameters. See :meth:`default_hparams` for the
            defaults.
    """

    def __init__(self, hparams):
        qDataBase.__init__(self, hparams)

        with tf.name_scope(self.name, self.default_hparams()["name"]):
            self._src_dataset, self._tgt_dataset = self.make_dataset(
                self._hparams.source_dataset, self._hparams.target_dataset)
            self._data_provider = self._make_data_provider(
                self._src_dataset, self._tgt_dataset)

    @staticmethod
    def default_hparams():
        """Returns a dicitionary of default hyperparameters.
        """
        hparams = qDataBase.default_hparams()
        hparams["name"] = "paired_text_data"
        hparams.update(_default_q_paired_text_dataset_hparams())
        return hparams

    @staticmethod
    def make_dataset(src_hparams, tgt_hparams):
        src_dataset = q_mono_text_data.qMonoTextData.make_dataset(
            src_hparams)
        src_dataset.decoder.text_tensor_name = "source_text"
        src_dataset.decoder.length_tensor_name = "source_length"
        src_dataset.decoder.text_id_tensor_name = "source_text_ids"

        if tgt_hparams["processing_share"]:
            tgt_proc_hparams = src_hparams["processing"]
        else:
            tgt_proc_hparams = tgt_hparams["processing"]

        # Make vocabulary
        bos_token = utils.default_string(tgt_proc_hparams["bos_token"],
                                         constants.BOS_TOKEN)
        eos_token = utils.default_string(tgt_proc_hparams["eos_token"],
                                         constants.EOS_TOKEN)
        if tgt_hparams["vocab_share"]:
            if bos_token == src_dataset.vocab.bos_token and \
                            eos_token == src_dataset.vocab.eos_token:
                tgt_vocab = src_dataset.vocab
            else:
                tgt_vocab = Vocab(src_hparams["vocab_file"],
                                  bos_token=bos_token,
                                  eos_token=eos_token)
        else:
            tgt_vocab = Vocab(tgt_hparams["vocab_file"],
                              bos_token=bos_token,
                              eos_token=eos_token)

        # Get reader class.
        if tgt_hparams["reader_share"]:
            tgt_reader_class = utils.get_class(src_hparams["reader"]["type"],
                                               ["tensorflow"])
        else:
            tgt_reader_class = utils.get_class(tgt_hparams["reader"]["type"],
                                               ["tensorflow"])

        # Get data decoder.
        tgt_decoder = TextDataDecoder(
            split_level=tgt_proc_hparams["split_level"],
            delimiter=tgt_proc_hparams["delimiter"],
            bos_token=tgt_proc_hparams["bos_token"],
            eos_token=tgt_proc_hparams["eos_token"],
            max_seq_length=tgt_proc_hparams["max_seq_length"],
            token_to_id_map=tgt_vocab.token_to_id_map,
            text_tensor_name="target_text",
            length_tensor_name="target_length",
            text_id_tensor_name="target_text_ids")

        # Load embedding (optional)
        if tgt_hparams["embedding_init_share"]:
            tgt_embedding = src_dataset.embedding
        else:
            emb_hparams = tgt_hparams["embedding_init"]
            tgt_embedding = None
            if emb_hparams["file"] is not None and emb_hparams["file"] != "":
                tgt_embedding = Embedding(
                    tgt_vocab.token_to_id_map_py, emb_hparams)

        # Create the dataset
        tgt_dataset = tf_slim.dataset.Dataset(
            data_sources=tgt_hparams["files"],
            reader=tgt_reader_class,
            decoder=tgt_decoder,
            num_samples=None,
            items_to_descriptions=None,
            vocab=tgt_vocab,
            embedding=tgt_embedding)

        return src_dataset, tgt_dataset

    def _make_data_provider(self, src_dataset, tgt_dataset):
        src_reader_kwargs = None
        if len(self._hparams.source_dataset["reader"]["kwargs"]) > 0:
            src_reader_kwargs = \
                self._hparams.source_dataset["reader"]["kwargs"].todict()

        tgt_reader_kwargs = None
        if self._hparams.target_dataset["reader_share"]:
            tgt_reader_kwargs = src_reader_kwargs
        elif len(self._hparams.target_dataset["reader"]["kwargs"]) > 0:
            tgt_reader_kwargs = \
                self._hparams.target_dataset["reader"]["kwargs"].todict()

        data_provider = ParallelDataProvider(
            datasets=[src_dataset, tgt_dataset],
            reader_kwargs=[src_reader_kwargs, tgt_reader_kwargs],
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
            input_length = tf.maximum(
                data[self._src_dataset.decoder.length_tensor_name],
                data[self._tgt_dataset.decoder.length_tensor_name]
            )
            #input_length = data[self._src_dataset.decoder.length_tensor_name]
            batch_size = self._hparams.bucket_batch_size
            if batch_size is None:
                batch_size = self._hparams.batch_size
            _, data_batch = tf.contrib.training.bucket_by_sequence_length(
                input_length=input_length,
                tensors=data,
                batch_size=batch_size,
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
    def source_dataset(self):
        """The `Dataset` instance representing the source dataset.
        """
        return self._src_dataset

    @property
    def target_dataset(self):
        """The `Dataset` instance representing the target dataset.
        """
        return self._tgt_dataset

    @property
    def source_vocab(self):
        """The :class:`~texar.data.Vocab` instance representing the vocabulary
        of the source dataset.
        """
        return self.source_dataset.vocab

    @property
    def target_vocab(self):
        """The :class:`~texar.data.Vocab` instance representing the vocabulary
        of the target dataset.
        """
        return self.target_dataset.vocab


