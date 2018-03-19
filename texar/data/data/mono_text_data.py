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
from texar.data.data import data_utils
from texar.data.data.text_data_base import TextDataBase
from texar.data.data_decoders import TextDataDecoder
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
        dataset = tf.data.TextLineDataset(
            self._hparams.dataset.files,
            compression_type=self._hparams.dataset.compression_type)
        return dataset

    def _process_dataset(self):
        # pylint: disable=attribute-defined-outside-init
        dataset_hparams = self._hparams.dataset

        # Create data decoder
        self._decoder = TextDataDecoder(
            delimiter=dataset_hparams["delimiter"],
            bos_token=dataset_hparams["bos_token"],
            eos_token=dataset_hparams["eos_token"],
            max_seq_length=dataset_hparams["max_seq_length"],
            token_to_id_map=self._vocab.token_to_id_map)

        # Create other transformations
        other_trans_hparams = dataset_hparams["other_transformations"]
        other_trans = []
        for tran in other_trans_hparams:
            if not utils.is_callable(tran):
                tran = utils.get_function(tran, ["texar.custom"])
            other_trans.append(data_utils.make_partial(tran, self))

        # Process data
        chained_tran = data_utils.make_chained_transformation(
            [self._decoder] + other_trans)
        num_parallel_calls = self._hparams.num_parallel_calls
        self._dataset = self._dataset.map(
            chained_tran, num_parallel_calls=num_parallel_calls)

    def _make_data(self):
        dataset_hparams = self._hparams.dataset
        self._vocab = self.make_vocab(dataset_hparams)
        self._embedding = self.make_embedding(
            dataset_hparams["embedding_init"], self._vocab.token_to_id_map_py)

        # Create dataset
        dataset = self._make_dataset()
        dataset, dataset_size = self._shuffle_dataset(
            dataset, self._hparams, self._hparams.dataset.files)
        self._dataset = dataset
        self._dataset_size = dataset_size

        # Processing.
        # Try to ensure all class attributes are created before this part,
        # so that the transformation func can have access to
        # them when called with `transformation_func(data, self)`
        self._process_dataset()

        # Batching
        length_func = lambda x: x[self._decoder.length_tensor_name]
        self._dataset = self._make_batch(
            self._dataset, self._hparams, length_func)

        if self._hparams.prefetch_buffer_size > 0:
            self._dataset = self._dataset.prefetch(
                self._hparams.prefetch_buffer_size)


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

    def dataset_size(self):
        """Returns the number of data instances in the dataset.
        """
        if not self._dataset_size:
            # pylint: disable=attribute-defined-outside-init
            self._dataset_size = data_utils.count_file_lines(
                self._hparams.dataset.files)
        return self._dataset_size

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

    @property
    def text_tensor_name(self):
        """The name of text tensor.
        """
        return self._decoder.text_tensor_name

    @property
    def length_tensor_name(self):
        """The name of length tensor.
        """
        return self._decoder.length_tensor_name

    @property
    def text_id_tensor_name(self):
        """The name of text index tensor.
        """
        return self._decoder.text_id_tensor_name

