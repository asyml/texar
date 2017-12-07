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
from texar.data.databases import mono_text_database
from texar.data.databases.database_base import DataBaseBase
from texar.data.databases.paired_text_database import PairedTextDataBase
from texar.data.databases.data_decoders import TextDataDecoder
from texar.data.databases.data_decoders import MultiSentenceTextDataDecoder
from texar.data.vocabulary import Vocab
from texar.data.embedding import Embedding

# pylint: disable=invalid-name, arguments-differ, not-context-manager
# pylint: disable=protected-access, no-member

__all__ = [
    "_default_multi_source_text_dataset_hparams",
    "MultiSourceTextDataBase"
]

def _default_multi_source_text_dataset_hparams():
    """Returns hyperparameters of a multi source text dataset with defaults.
    """
    source_hparams = mono_text_database._default_mono_text_dataset_hparams()
    source_hparams["processing"]["bos_token"] = None
    source_hparams["processing"]["max_context_length"] = 5
    target_hparams = mono_text_database._default_mono_text_dataset_hparams()
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


class MultiSourceTextDataBase(PairedTextDataBase):
    """Text data base that reads pair of data. The only difference is that
    each line the source data is assumed to be concatenation of multiple
    sentences.

    This database is suitable for Dialog models.
    """

    def __init__(self, hparams):
        #TODO(zhiting): Should call PairedTextDataBase.__init__ ?
        DataBaseBase.__init__(self, hparams)

        with tf.name_scope(self.name, self.default_hparams()["name"]):
            self._src_dataset, self._tgt_dataset = self.make_dataset(
                self._hparams.source_dataset, self._hparams.target_dataset)
            self._data_provider = self._make_data_provider(
                self._src_dataset, self._tgt_dataset)

    @staticmethod
    def default_hparams():
        """Returns a dicitionary of default hyperparameters.
        """
        hparams = DataBaseBase.default_hparams()
        hparams["name"] = "multi_source_text_database"
        hparams.update(_default_multi_source_text_dataset_hparams())
        return hparams

    @staticmethod
    def make_source_dataset(dataset_hparams):
        proc_hparams = dataset_hparams["processing"]

        # Load vocabulary
        bos_token = utils.default_string(proc_hparams["bos_token"], "<BOS>")
        eos_token = utils.default_string(proc_hparams["eos_token"], "<EOS>")
        vocab = Vocab(dataset_hparams["vocab_file"],
                      bos_token=bos_token,
                      eos_token=eos_token)

        # Get the reader class
        reader_class = utils.get_class(dataset_hparams["reader"]["type"],
                                       ["tensorflow"])

        # Create a multi sentence data decoder with default sentence delimiter.
        src_decoder = MultiSentenceTextDataDecoder(
            split_level=proc_hparams["split_level"],
            delimiter=proc_hparams["delimiter"],
            bos_token=proc_hparams["bos_token"],
            eos_token=proc_hparams["eos_token"],
            max_seq_length=proc_hparams["max_seq_length"],
            max_context_length=proc_hparams["max_context_length"],
            token_to_id_map=vocab.token_to_id_map)

        # Load embedding (optional)
        emb_hparams = dataset_hparams["embedding_init"]
        embedding = None
        if emb_hparams["file"] is not None and emb_hparams["file"] != "":
            embedding = Embedding(vocab.token_to_id_map_py, emb_hparams)

        # Create the dataset
        dataset = tf_slim.dataset.Dataset(
            data_sources=dataset_hparams["files"],
            reader=reader_class,
            decoder=src_decoder,
            num_samples=None,
            items_to_descriptions=None,
            vocab=vocab,
            embedding=embedding)

        return dataset

    @staticmethod
    def make_dataset(src_hparams, tgt_hparams):
        src_dataset = MultiSourceTextDataBase.make_source_dataset(src_hparams)
        src_dataset.decoder.text_tensor_name = "source_text"
        src_dataset.decoder.length_tensor_name = "source_length"
        src_dataset.decoder.text_id_tensor_name = "source_text_ids"

        if tgt_hparams["processing_share"]:
            tgt_proc_hparams = src_hparams["processing"]
        else:
            tgt_proc_hparams = tgt_hparams["processing"]

        # Make vocabulary
        bos_token = utils.default_string(tgt_proc_hparams["bos_token"],
                                         "<BOS>")
        eos_token = utils.default_string(tgt_proc_hparams["eos_token"],
                                         "<EOS>")
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

    def __call__(self):
        data = self._data_provider.get(self.list_items())
        data = dict(zip(self.list_items(), data))

        num_threads = 1
        # Recommended capacity =
        # (num_threads + a small safety margin) * batch_size + margin
        capacity = (num_threads + 32) * self._hparams.batch_size + 1024

        allow_smaller_final_batch = self._hparams.allow_smaller_final_batch

        data_batch = tf.train.batch(
            tensors=data,
            batch_size=self._hparams.batch_size,
            num_threads=num_threads,
            capacity=capacity,
            enqueue_many=False,
            dynamic_pad=True,
            allow_smaller_final_batch=allow_smaller_final_batch,
            name="%s/batch" % self.name)

        return data_batch
