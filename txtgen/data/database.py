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

from txtgen.hyperparams import HParams
from txtgen.core import utils
from txtgen.data.text_data_decoder import TextDataDecoder, \
    MultiSentenceTextDataDecoder
from txtgen.data.data_providers import PairedDataProvider
from txtgen.data.vocabulary import Vocab
from txtgen.data.embedding import Embedding


# pylint: disable=invalid-name, arguments-differ

def _default_mono_text_dataset_hparams():
    """Returns hyperparameters of a mono text dataset with default values.
    """
    # TODO(zhiting): add more docs
    return {
        "files": [],
        "vocab_file": "",
        "embedding_init": Embedding.default_hparams(),
        "reader": {
            "type": "tensorflow.TextLineReader",
            "kwargs": {}
        },
        "processing": {
            "split_level": "word",
            "delimiter": " ",
            "max_seq_length": None,
            "bos_token": "<BOS>",
            "eos_token": "<EOS>"
        }
    }


def _default_paired_text_dataset_hparams():
    """Returns hyperparameters of a paired text dataset with default values.
    """
    source_hparams = _default_mono_text_dataset_hparams()
    source_hparams["processing"]["bos_token"] = None
    target_hparams = _default_mono_text_dataset_hparams()
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


def _default_multi_source_text_dataset_hparams():
    """Returns hyperparameters of a multi source text dataset with defaults.
    """
    source_hparams = _default_mono_text_dataset_hparams()
    source_hparams["processing"]["bos_token"] = None
    source_hparams["processing"]["max_context_length"] = 5
    target_hparams = _default_mono_text_dataset_hparams()
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


class DataBaseBase(object):
    """Base class of all data classes.
    """

    def __init__(self, hparams):
        self._hparams = HParams(hparams, self.default_hparams())

    # TODO (zhiting): add more docs
    @staticmethod
    def default_hparams():
        """Returns a dictionary of default hyperparameters.
        """
        return {
            "name": "database",
            "num_epochs": None,
            "batch_size": 64,
            "allow_smaller_final_batch": False,
            "bucket_boundaries": [],
            "shuffle": True,
            "seed": None
        }

    @staticmethod
    def make_dataset(dataset_hparams):
        """Creates a Dataset instance that defines source filenames, data
        reading and decoding methods, vocabulary, and embedding initial
        values.

        Args:
            dataset_hparams (dict or HParams): Dataset hyperparameters.
        """
        raise NotImplementedError

    def _make_data_provider(self, dataset):
        """Creates a DataProvider instance that provides a single example of
        requested data.

        Args:
            dataset (Dataset): The dataset used to provide data examples.
        """
        raise NotImplementedError

    def __call__(self):
        """Returns a batch of data.
        """
        raise NotImplementedError

    @property
    def batch_size(self):
        """An integer representing the batch size.
        """
        return self._hparams.batch_size

    @property
    def hparams(self):
        """A :class:`~txtgen.hyperparams.HParams` instance of the
        database hyperparameters.
        """
        return self._hparams

    @property
    def name(self):
        """The name of the data base.
        """
        return self.hparams.name


# pylint: disable=no-member

class MonoTextDataBase(DataBaseBase):
    """Text data base that reads single set of text files.

    This is for the use of, e.g., language models, auto-encoders, etc. For
    models that involve two sets of text files (`source` and `target`), use
    :class:`~txtgen.data.database.PairedTextDataBase`.

    Args:
        hparams (dict): Hyperparameters. See :meth:`default_hparams` for the
            defaults.
    """

    def __init__(self, hparams):
        DataBaseBase.__init__(self, hparams)

        # pylint: disable=not-context-manager
        with tf.name_scope(self.name, self.default_hparams()["name"]):
            self._dataset = self.make_dataset(self._hparams.dataset)
            self._data_provider = self._make_data_provider(self._dataset)

    @staticmethod
    def default_hparams():
        """Returns a dicitionary of default hyperparameters.
        """
        hparams = DataBaseBase.default_hparams()
        hparams["name"] = "mono_text_database"
        hparams.update({
            "dataset": _default_mono_text_dataset_hparams()
        })
        return hparams

    @staticmethod
    def make_dataset(dataset_hparams):
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

        # Create data decoder
        decoder = TextDataDecoder(
            split_level=proc_hparams["split_level"],
            delimiter=proc_hparams["delimiter"],
            bos_token=proc_hparams["bos_token"],
            eos_token=proc_hparams["eos_token"],
            max_seq_length=proc_hparams["max_seq_length"],
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
            decoder=decoder,
            num_samples=None,
            items_to_descriptions=None,
            vocab=vocab,
            embedding=embedding)

        return dataset

    def _make_data_provider(self, dataset):
        reader_kwargs = None
        if len(self._hparams.dataset["reader"]["kwargs"]) > 0:
            reader_kwargs = self._hparams.dataset["reader"]["kwargs"].todict()
        data_provider = tf_slim.dataset_data_provider.DatasetDataProvider(
            dataset=dataset,
            num_readers=1,  # TODO(zhiting): allow more readers
            reader_kwargs=reader_kwargs,
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
            input_length = data[self._dataset.decoder.length_tensor_name]
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
        items = list(self._data_provider.list_items())
        # Discard extra tensors inserted by DatasetDataProvider
        if 'record_key' in items:
            items.remove('record_key')
        return items

    @property
    def dataset(self):
        """The dataset.
        """
        return self._dataset

    @property
    def vocab(self):
        """The vocabulary defined in :class:`~txtgen.data.Vocab`.
        """
        return self.dataset.vocab


class PairedTextDataBase(DataBaseBase):
    """Text data base that reads source and target text.

    This is for the use of, e.g., seq2seq models.

    Args:
        hparams (dict): Hyperparameters. See :meth:`default_hparams` for the
            defaults.
    """

    def __init__(self, hparams):
        DataBaseBase.__init__(self, hparams)

        # pylint: disable=not-context-manager
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
        hparams["name"] = "paired_text_database"
        hparams.update(_default_paired_text_dataset_hparams())
        return hparams

    @staticmethod
    def make_dataset(src_hparams, tgt_hparams):
        src_dataset = MonoTextDataBase.make_dataset(src_hparams)
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

        data_provider = PairedDataProvider(
            dataset1=src_dataset,
            dataset2=tgt_dataset,
            reader_kwargs1=src_reader_kwargs,
            reader_kwargs2=tgt_reader_kwargs,
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
        """The :class:`~txtgen.data.Vocab` instance representing the vocabulary
        of the source dataset.
        """
        return self.source_dataset.vocab

    @property
    def target_vocab(self):
        """The :class:`~txtgen.data.Vocab` instance representing the vocabulary
        of the target dataset.
        """
        return self.target_dataset.vocab


class MultiSourceTextDataBase(PairedTextDataBase):
    """    Text data base that reads pair of data. The only difference is that
    each line the source data is assumed to be concatenation of multiple
    sentences.

       This database is suitable for Dialog models.
    """

    def __init__(self, hparams):
        DataBaseBase.__init__(self, hparams)

        # pylint: disable=not-context-manager
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
