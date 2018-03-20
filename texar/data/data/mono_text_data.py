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

    @staticmethod
    def _make_mono_text_dataset(dataset_hparams):
        dataset = tf.data.TextLineDataset(
            dataset_hparams["files"],
            compression_type=dataset_hparams["compression_type"])
        return dataset

    @staticmethod
    def _make_other_transformations(other_trans_hparams, text_data):
        """Creates a list of tranformation functions based on the
        hyperparameters.

        Args:
            other_trans_hparams (list): A list of transformation functions,
                names, or full paths.
            text_data: An instance of :class:`texar.data.TextDataBase` to
                be passed to transformation functions.

        Returns:
            A list of transformation functions.
        """
        other_trans = []
        for tran in other_trans_hparams:
            if not utils.is_callable(tran):
                tran = utils.get_function(tran, ["texar.custom"])
            other_trans.append(data_utils.make_partial(tran, text_data))
        return other_trans

    @staticmethod
    def _make_processor(dataset_hparams, data_spec):
        # Create data decoder
        decoder = TextDataDecoder(
            delimiter=dataset_hparams["delimiter"],
            bos_token=dataset_hparams["bos_token"],
            eos_token=dataset_hparams["eos_token"],
            max_seq_length=dataset_hparams["max_seq_length"],
            token_to_id_map=data_spec.vocab.token_to_id_map)

        # Create other transformations
        data_spec.add_spec(decoder=decoder)
        other_trans = MonoTextData._make_other_transformations(
            dataset_hparams["other_transformations"], data_spec)

        # Process data
        chained_tran = data_utils.make_chained_transformation(
            [decoder] + other_trans)

        return chained_tran, data_spec

    @staticmethod
    def _process_dataset(dataset, hparams, data_spec):
        chained_tran, data_spec = MonoTextData._make_processor(
            hparams["dataset"], data_spec)
        num_parallel_calls = hparams["num_parallel_calls"]
        dataset = dataset.map(
            lambda *args: chained_tran(data_utils.maybe_tuple(args)),
            num_parallel_calls=num_parallel_calls)
        return dataset, data_spec

    def _make_length_fn(self):
        length_fn = self._hparams.bucket_length_fn
        if not length_fn:
            length_fn = lambda x: x[self._decoder.length_tensor_name]
        elif not utils.is_callable(length_fn):
            # pylint: disable=redefined-variable-type
            length_fn = utils.get_function(length_fn, ["texar.custom"])
        return length_fn

    def _make_data(self):
        dataset_hparams = self._hparams.dataset
        self._vocab = self.make_vocab(dataset_hparams)
        self._embedding = self.make_embedding(
            dataset_hparams["embedding_init"], self._vocab.token_to_id_map_py)

        # Create and shuffle dataset
        dataset = self._make_mono_text_dataset(dataset_hparams)
        dataset, dataset_size = self._shuffle_dataset(
            dataset, self._hparams, self._hparams.dataset.files)
        self._dataset_size = dataset_size

        # Processing
        # pylint: disable=protected-access
        data_spec = data_utils._DataSpec(dataset=dataset,
                                         dataset_size=self._dataset_size,
                                         vocab=self._vocab,
                                         embedding=self._embedding)
        dataset, data_spec = self._process_dataset(dataset, self._hparams,
                                                   data_spec)
        self._data_spec = data_spec

        # Batching
        length_fn = self._make_length_fn()
        dataset = self._make_batch(dataset, self._hparams, length_fn)

        # Prefetching
        if self._hparams.prefetch_buffer_size > 0:
            dataset = dataset.prefetch(self._hparams.prefetch_buffer_size)

        self._dataset = dataset

    def list_items(self):
        """Returns the list of item names that the data can produce.

        Returns:
            A list of strings.
        """
        return list(self._dataset.output_types.keys())

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
    def text_name(self):
        """The name of text tensor.
        """
        return self._decoder.text_tensor_name

    @property
    def length_name(self):
        """The name of length tensor.
        """
        return self._decoder.length_tensor_name

    @property
    def text_id_name(self):
        """The name of text index tensor.
        """
        return self._decoder.text_id_tensor_name

