#
"""
Various data classes that define data reading, parsing, batching, and other
preprocessing operations.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf

from texar.core import utils
from texar.data.data.text_data_base import TextDataBase
from texar.data.data_decoders import TextDataDecoder
from texar.data.data import data_utils
from texar.data.vocabulary import Vocab
from texar.data.embedding import Embedding
from texar.data.constants import BOS_TOKEN, EOS_TOKEN
# pylint: disable=invalid-name, arguments-differ, not-context-manager

__all__ = [
    "_default_mono_text_dataset_hparams",
    "MonoTextData"
]

def _default_mono_text_dataset_hparams():
    """Returns hyperparameters of a mono text dataset with default values.
    """
    # TODO(zhiting): add more docs
    return {
        "files": [],
        "compression_type": None,
        "vocab_file": "",
        "embedding_init": Embedding.default_hparams(),
        "delimiter": " ",
        "max_seq_length": None,
        "bos_token": BOS_TOKEN,
        "eos_token": EOS_TOKEN,
        "other_transformations": [],
        "@no_typecheck": ["files"]
    }

# pylint: disable=no-member

class MonoTextData(TextDataBase):
    """Text data base that reads single set of text files.

    This is for the use of, e.g., language models, auto-encoders, etc. For
    models that involve two sets of text files (`source` and `target`), use
    :class:`~texar.data.database.PairedTextDataBase`.

    Args:
        hparams (dict): Hyperparameters. See :meth:`default_hparams` for the
            defaults.
    """

    def __init__(self, hparams):
        TextDataBase.__init__(self, hparams)
        with tf.name_scope(self.name, self.default_hparams()["name"]):
            self._make_data()

    @staticmethod
    def default_hparams():
        """Returns a dicitionary of default hyperparameters.
        """
        hparams = TextDataBase.default_hparams()
        hparams["name"] = "mono_text_data"
        hparams.update({
            "dataset": _default_mono_text_dataset_hparams()
        })
        return hparams

    @staticmethod
    def make_vocab(hparams):
        """Reads vocab file and returns an instance of
        :class:`texar.data.Vocab`.
        """
        bos_token = utils.default_string(hparams["bos_token"], BOS_TOKEN)
        eos_token = utils.default_string(hparams["eos_token"], EOS_TOKEN)
        vocab = Vocab(hparams["vocab_file"],
                      bos_token=bos_token, eos_token=eos_token)
        return vocab

    @staticmethod
    def make_embedding(emb_hparams, token_to_id_map):
        """Optionally loads embedding from file (if provided), and returns
        an instance of :class:`texar.data.Embedding`.
        """
        embedding = None
        if emb_hparams["file"] is not None and len(emb_hparams["file"]) > 0:
            embedding = Embedding(token_to_id_map, emb_hparams)
        return embedding

    def _make_dataset(self):
        dataset = tf.data.TextLineDataset(self._hparams.dataset.files)
        return dataset

    @staticmethod
    def _shuffle_dataset(dataset, hparams, dataset_files):
        shuffle_buffer_size = hparams["shuffle_buffer_size"]
        if hparams["shard_and_shuffle"]:
            if shuffle_buffer_size is None:
                raise ValueError(
                    "Dataset hyperparameter 'shuffle_buffer_size' "
                    "must not be `None` if 'shard_and_shuffle'=`True`.")
            dataset_size = data_utils.count_file_lines(dataset_files)
            if shuffle_buffer_size >= dataset_size:
                raise ValueError(
                    "Dataset size (%d) <= shuffle_buffer_size (%d). Set "
                    "shuffle_and_shard to `False`." %
                    (dataset_size, shuffle_buffer_size))
            #TODO(zhiting): Use a different seed?
            dataset = dataset.apply(data_utils.random_shard_dataset(
                dataset_size, shuffle_buffer_size, hparams["seed"]))
            dataset = dataset.shuffle(shuffle_buffer_size + 16, # add a margin
                                      seed=hparams["seed"])
        elif hparams["shuffle"]:
            if shuffle_buffer_size is None:
                shuffle_buffer_size = data_utils.count_file_lines(dataset_files)
            dataset = dataset.shuffle(shuffle_buffer_size, seed=hparams["seed"])

        return dataset

    def _process_dataset(self, dataset):
        dataset_hparams = self._hparams.dataset

        # Create data decoder
        decoder = TextDataDecoder(
            delimiter=dataset_hparams["delimiter"],
            bos_token=dataset_hparams["bos_token"],
            eos_token=dataset_hparams["eos_token"],
            max_seq_length=dataset_hparams["max_seq_length"],
            token_to_id_map=self._vocab.token_to_id_map)

        # Process data
        num_parallel_calls = self._hparams.num_parallel_calls
        dataset = dataset.map(
            decoder, num_parallel_calls=num_parallel_calls)

        other_trans = dataset_hparams["other_transformations"]
        if len(other_trans) > 0:
            for tran in other_trans:
                tran_fn = utils.get_function(tran, ["texar.custom"])
                other_trans.append(tran_fn)
                dataset = dataset.map(
                    lambda x: other_trans[-1](x, self),
                    num_parallel_calls=num_parallel_calls)

        return dataset, decoder

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
            bucket_batch_size = dataset["bucket_batch_sizes"]
            if bucket_batch_size is None:
                bucket_batch_size = [batch_size] * (len(bucket_boundaries) + 1)
            dataset = tf.contrib.data.bucket_by_sequence_length(
                element_length_func, bucket_boundaries, bucket_batch_size)
        return dataset

    def _make_data(self):
        dataset_hparams = self._hparams.dataset
        self._vocab = self.make_vocab(dataset_hparams)
        self._embedding = self.make_embedding(
            dataset_hparams["embedding_init"], self._vocab.token_to_id_map_py)

        # Create dataset
        dataset = self._make_dataset()
        dataset = self._shuffle_dataset(
            dataset, self._hparams, self._hparams.dataset.files)

        # Processing
        dataset, self._decoder = self._process_dataset(dataset)

        # Batching
        length_func = lambda x: x[self._decoder.length_tensor_name]
        dataset = self._make_batch(
            dataset, self._hparams, length_func)

        if self._hparams.prefetch_buffer_size > 0:
            dataset = dataset.prefetch(self._hparams.prefetch_buffer_size)

        self._dataset = dataset

    def list_items(self):
        """Returns the list of item names that the data can produce.

        Returns:
            A list of strings.
        """
        return list(self._decoder.list_items())

    @property
    def dataset(self):
        """The dataset.
        """
        return self._dataset

    @property
    def vocab(self):
        """The vocabulary defined in :class:`~texar.data.Vocab`.
        """
        return self._vocab

    @property
    def embedding_init_value(self):
        """The `Tensor` containing the embedding value loaded from file.
        `None` if embedding is not specified.
        """
        if self._embedding is None:
            return None
        return self._embedding.word_vecs

