#
"""
Paired text data that consists of source text and target text.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf

from texar.core import utils
from texar.data import data
from texar.data.data.text_data_base import TextDataBase
from texar.data.data.mono_text_data import MonoTextData
from texar.data.data_decoders import TextDataDecoder
from texar.data.data import data_utils
from texar.data.vocabulary import Vocab
from texar.data.embedding import Embedding
from texar.data.constants import BOS_TOKEN, EOS_TOKEN

# pylint: disable=invalid-name, arguments-differ, not-context-manager
# pylint: disable=protected-access

__all__ = [
    "_default_paired_text_dataset_hparams",
    "PairedTextData"
]

def _default_paired_text_dataset_hparams():
    """Returns hyperparameters of a mono text dataset with default values.
    """
    # TODO(zhiting): add more docs
    source_hparams = data.mono_text_data._default_mono_text_dataset_hparams()
    source_hparams["bos_token"] = None
    target_hparams = data.mono_text_data._default_mono_text_dataset_hparams()
    target_hparams.update(
        {
            "vocab_share": False,
            "embedding_init_share": False,
            "processing_share": False
        }
    )
    return {
        "source_dataset": source_hparams,
        "target_dataset": target_hparams
    }


class PairedTextData(TextDataBase):
    """Text data base that reads source and target text.

    This is for the use of, e.g., seq2seq models.

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
        hparams["name"] = "paired_text_data"
        hparams.update(_default_paired_text_dataset_hparams())
        return hparams

    @staticmethod
    def make_vocab(src_hparams, tgt_hparams):
        """Reads vocab files and returns source vocab and target vocab.

        Args:
            src_hparams (dict or HParams): Hyperparameters of source dataset.
            tgt_hparams (dict or HParams): Hyperparameters of target dataset.

        Returns:
            A pair of :class:`texar.data.Vocab` instances. The two instances
            may be the same objects if source and target vocabs are shared
            and have the same other configs.
        """
        src_vocab = MonoTextData.make_vocab(src_hparams)

        if tgt_hparams["processing_share"]:
            tgt_bos_token = src_hparams["bos_token"]
            tgt_eos_token = src_hparams["eos_token"]
        else:
            tgt_bos_token = tgt_hparams["bos_token"]
            tgt_eos_token = tgt_hparams["eos_token"]
        tgt_bos_token = utils.default_string(tgt_bos_token,
                                             BOS_TOKEN)
        tgt_eos_token = utils.default_string(tgt_eos_token,
                                             EOS_TOKEN)
        if tgt_hparams["vocab_share"]:
            if tgt_bos_token == src_vocab.bos_token and \
                    tgt_eos_token == src_vocab.eos_token:
                tgt_vocab = src_vocab
            else:
                tgt_vocab = Vocab(src_hparams["vocab_file"],
                                  bos_token=tgt_bos_token,
                                  eos_token=tgt_eos_token)
        else:
            tgt_vocab = Vocab(tgt_hparams["vocab_file"],
                              bos_token=tgt_bos_token,
                              eos_token=tgt_eos_token)

        return src_vocab, tgt_vocab


    @staticmethod
    def make_embedding(src_emb_hparams, src_token_to_id_map,
                       tgt_emb_hparams=None, tgt_token_to_id_map=None,
                       emb_init_share=False):
        """Optionally loads source and target embeddings from files
        (if provided), and returns respective :class:`texar.data.Embedding`
        instances.
        """
        src_embedding = MonoTextData.make_embedding(src_emb_hparams,
                                                    src_token_to_id_map)
        #if not vocab_share and emb_init_share:
        #    raise ValueError("embedding init can be shared only when vocab "
        #                     "is shared. Got `vocab_share=False, "
        #                     "emb_init_share=True`.")

        if emb_init_share:
            tgt_embedding = src_embedding
        else:
            tgt_emb_file = tgt_emb_hparams["file"]
            tgt_embedding = None
            if tgt_emb_file is not None and tgt_emb_file != "":
                tgt_embedding = Embedding(tgt_token_to_id_map, tgt_emb_hparams)

        return src_embedding, tgt_embedding

    def _make_dataset(self):
        src_dataset = tf.data.TextLineDataset(
            self._hparams.source_dataset.files,
            compression_type=self._hparams.source_dataset.compression_type)
        tgt_dataset = tf.data.TextLineDataset(
            self._hparams.target_dataset.files,
            compression_type=self._hparams.target_dataset.compression_type)
        return tf.data.Dataset.zip((src_dataset, tgt_dataset))

    def _process_dataset(self, dataset):
        # Create source data decoder
        src_hparams = self._hparams.source_dataset
        src_decoder = TextDataDecoder(
            delimiter=src_hparams["delimiter"],
            bos_token=src_hparams["bos_token"],
            eos_token=src_hparams["eos_token"],
            max_seq_length=src_hparams["max_seq_length"],
            token_to_id_map=self._src_vocab.token_to_id_map,
            text_tensor_name="source_text",
            length_tensor_name="source_length",
            text_id_tensor_name="source_text_ids")
        # Create target data decoder
        tgt_hparams = self._hparams.source_dataset
        tgt_decoder = TextDataDecoder(
            delimiter=tgt_hparams["delimiter"],
            bos_token=tgt_hparams["bos_token"],
            eos_token=tgt_hparams["eos_token"],
            max_seq_length=tgt_hparams["max_seq_length"],
            token_to_id_map=self._tgt_vocab.token_to_id_map,
            text_tensor_name="target_text",
            length_tensor_name="target_length",
            text_id_tensor_name="target_text_ids")
        # Process data
        tran_fn = data_utils.make_combined_transformation([src_decoder,
                                                           tgt_decoder])
        num_parallel_calls = self._hparams.num_parallel_calls
        dataset = dataset.map(
            tran_fn, num_parallel_calls=num_parallel_calls)

        return dataset, src_decoder, tgt_decoder

    def _perform_other_transformations(self, dataset):
        num_parallel_calls = self._hparams.num_parallel_calls

        src_trans_hparams = self._hparams.source_dataset.other_transformations
        src_trans = []
        for tran in src_trans_hparams:
            if not utils.is_callable(tran):
                tran = utils.get_function(tran, ["texar.custom"])
            src_trans.append(tran)

        tgt_trans_hparams = self._hparams.target_dataset.other_transformations
        tgt_trans = []
        for tran in tgt_trans_hparams:
            if not utils.is_callable(tran):
                tran = utils.get_function(tran, ["texar.custom"])
            tgt_trans.append(tran)

        tran_fn = data_utils.make_combined_transformation(
            [src_trans, tgt_trans], self)

        dataset = dataset.map(
            tran_fn, num_parallel_calls=num_parallel_calls)

        return dataset

    def _make_data(self):
        self._src_vocab, self._tgt_vocab = self.make_vocab(
            self._hparams.source_dataset, self._hparams.target_dataset)
        self._embedding = self.make_embedding(
            self._hparams.source_dataset.embedding_init,
            self._src_vocab.token_to_id_map_py,
            self._hparams.target_dataset.embedding_init,
            self._tgt_vocab.token_to_id_map_py,
            self._hparams.target_dataset.embedding_init_share)

        # Create dataset
        dataset = self._make_dataset()
        dataset, dataset_size = self._shuffle_dataset(
            dataset, self._hparams, self._hparams.source_dataset.files)
        self._dataset_size = dataset_size

        # Processing
        self._dataset, self._src_decoder, self._tgt_decoder = \
            self._process_dataset(dataset)
        # Try to ensure all class attributes are created before this part,
        # so that the transformation func can have access to
        # them when called with `transformation_func(data, self)`
        self._dataset = self._perform_other_transformations(self._dataset)
        # Transform data tuple into dict
        self._dataset =

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

