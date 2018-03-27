#
"""
Mono text data class that define data reading, parsing, batching, and other
preprocessing operations.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf

from texar.utils import utils
from texar.data.data import dataset_utils as dsutils
from texar.data.data.text_data_base import TextDataBase
from texar.data.data_decoders import TextDataDecoder, VarUttTextDataDecoder
from texar.data.vocabulary import Vocab, _SpecialTokens
from texar.data.embedding import Embedding

# pylint: disable=invalid-name, arguments-differ, protected-access

__all__ = [
    "_default_mono_text_dataset_hparams",
    "MonoTextData"
]

class _LengthFilterMode: # pylint: disable=old-style-class, no-init, too-few-public-methods
    """Options of length filter mode.
    """
    TRUNC = "truncate"
    DISCARD = "discard"

def _default_mono_text_dataset_hparams():
    """Returns hyperparameters of a mono text dataset with default values.

    Returns:
        .. code-block:: python

            {
            }

        Here:

        "max_seq_length" : int, optional
            Maximum length of output sequences. Data samples exceeding the
            length will be truncated or discarded according to
            `"length_filter_mode"`. The length does not include any added
            `"bos_token"` or `"eos_token"`. If `None` (default), no filtering
            is performed.

        "length_filter_mode" : str
            Either `"truncate"` or `"discard"`. If `"truncate"` (default),
            tokens exceeding the `"max_seq_length"` will be truncated.
            If `"discard"`, data samples longer than the `"max_seq_length"`
            will be discarded.

    """
    # TODO(zhiting): add more docs
    return {
        "files": [],
        "compression_type": None,
        "vocab_file": "",
        "embedding_init": Embedding.default_hparams(),
        "delimiter": " ",
        "max_seq_length": None,
        "length_filter_mode": "truncate",
        "bos_token": _SpecialTokens.BOS_TOKEN,
        "eos_token": _SpecialTokens.EOS_TOKEN,
        "other_transformations": [],
        "variable_utterance": False,
        "max_utterance_cnt": 5,
        "data_name": None,
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
        bos_token = utils.default_string(
            hparams["bos_token"], _SpecialTokens.BOS_TOKEN)
        eos_token = utils.default_string(
            hparams["eos_token"], _SpecialTokens.EOS_TOKEN)
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
    def _make_other_transformations(other_trans_hparams, data_spec):
        """Creates a list of tranformation functions based on the
        hyperparameters.

        Args:
            other_trans_hparams (list): A list of transformation functions,
                names, or full paths.
            data_spec: An instance of :class:`texar.data._DataSpec` to
                be passed to transformation functions.

        Returns:
            A list of transformation functions.
        """
        other_trans = []
        for tran in other_trans_hparams:
            if not utils.is_callable(tran):
                tran = utils.get_function(tran, ["texar.custom"])
            other_trans.append(dsutils.make_partial(tran, data_spec))
        return other_trans

    @staticmethod
    def _make_processor(dataset_hparams, data_spec, chained=True,
                        name_prefix=None):
        # Create data decoder
        max_seq_length = None
        if dataset_hparams["length_filter_mode"] == "truncate":
            max_seq_length = dataset_hparams["max_seq_length"]

        if not dataset_hparams["variable_utterance"]:
            decoder = TextDataDecoder(
                delimiter=dataset_hparams["delimiter"],
                bos_token=dataset_hparams["bos_token"],
                eos_token=dataset_hparams["eos_token"],
                max_seq_length=max_seq_length,
                token_to_id_map=data_spec.vocab.token_to_id_map)
        else:
            decoder = VarUttTextDataDecoder( # pylint: disable=redefined-variable-type
                delimiter=dataset_hparams["delimiter"],
                bos_token=dataset_hparams["bos_token"],
                eos_token=dataset_hparams["eos_token"],
                max_seq_length=max_seq_length,
                max_utterance_cnt=dataset_hparams["max_utterance_cnt"],
                token_to_id_map=data_spec.vocab.token_to_id_map)

        # Create other transformations
        data_spec.add_spec(decoder=decoder)
        other_trans = MonoTextData._make_other_transformations(
            dataset_hparams["other_transformations"], data_spec)
        if name_prefix:
            other_trans.append(dsutils.name_prefix_fn(name_prefix))

        data_spec.add_spec(name_prefix=name_prefix)

        if chained:
            chained_tran = dsutils.make_chained_transformation(
                [decoder] + other_trans)
            return chained_tran, data_spec
        else:
            return decoder, other_trans, data_spec

    @staticmethod
    def _make_length_filter(dataset_hparams, length_name, decoder):
        filter_mode = dataset_hparams["length_filter_mode"]
        max_length = dataset_hparams["max_seq_length"]
        filter_fn = None
        if filter_mode == _LengthFilterMode.DISCARD and max_length is not None:
            max_length += decoder.added_length
            filter_fn = dsutils._make_length_filter_fn(length_name,
                                                       max_length)
        return filter_fn

    def _process_dataset(self, dataset, hparams, data_spec):
        chained_tran, data_spec = self._make_processor(
            hparams["dataset"], data_spec,
            name_prefix=hparams["dataset"]["data_name"])
        num_parallel_calls = hparams["num_parallel_calls"]
        dataset = dataset.map(
            lambda *args: chained_tran(dsutils.maybe_tuple(args)),
            num_parallel_calls=num_parallel_calls)

        # Filter by length
        length_name = dsutils._connect_name(
            data_spec.name_prefix,
            data_spec.decoder.length_tensor_name)
        filter_fn = self._make_length_filter(
            hparams["dataset"], length_name, data_spec.decoder)
        if filter_fn:
            dataset = dataset.apply(lambda dataset: dataset.filter(filter_fn))

        return dataset, data_spec

    def _make_bucket_length_fn(self):
        length_fn = self._hparams.bucket_length_fn
        if not length_fn:
            length_fn = lambda x: x[self.length_name]
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
        data_spec = dsutils._DataSpec(dataset=dataset,
                                      dataset_size=self._dataset_size,
                                      vocab=self._vocab,
                                      embedding=self._embedding)
        dataset, data_spec = self._process_dataset(dataset, self._hparams,
                                                   data_spec)
        self._data_spec = data_spec
        self._decoder = data_spec.decoder

        # Batching
        length_fn = self._make_bucket_length_fn()
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
            self._dataset_size = dsutils.count_file_lines(
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
        name = dsutils._connect_name(
            self._data_spec.name_prefix,
            self._data_spec.decoder.text_tensor_name)
        return name

    @property
    def length_name(self):
        """The name of length tensor.
        """
        name = dsutils._connect_name(
            self._data_spec.name_prefix,
            self._data_spec.decoder.length_tensor_name)
        return name

    @property
    def text_id_name(self):
        """The name of text index tensor.
        """
        name = dsutils._connect_name(
            self._data_spec.name_prefix,
            self._data_spec.decoder.text_id_tensor_name)
        return name

    @property
    def utterance_cnt_name(self):
        """The name of utterance count tensor.
        """
        if not self._hparams.dataset.variable_utterance:
            raise ValueError("`utterance_cnt_name` is not defined.")
        name = dsutils._connect_name(
            self._data_spec.name_prefix,
            self._data_spec.decoder.utterance_cnt_tensor_name)
        return name

