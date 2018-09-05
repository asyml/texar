# Copyright 2018 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Paired text data that consists of source text and target text.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy

import tensorflow as tf

from texar.utils import utils
from texar.utils.dtypes import is_callable
from texar.data.data.mono_text_data import _default_mono_text_dataset_hparams
from texar.data.data.text_data_base import TextDataBase
from texar.data.data.mono_text_data import MonoTextData
from texar.data.data_utils import count_file_lines
from texar.data.data import dataset_utils as dsutils
from texar.data.vocabulary import Vocab, SpecialTokens
from texar.data.embedding import Embedding

# pylint: disable=invalid-name, arguments-differ, not-context-manager
# pylint: disable=protected-access, too-many-arguments

__all__ = [
    "_default_paired_text_dataset_hparams",
    "PairedTextData"
]

def _default_paired_text_dataset_hparams():
    """Returns hyperparameters of a paired text dataset with default values.

    See :meth:`texar.data.PairedTextData.default_hparams` for details.
    """
    source_hparams = _default_mono_text_dataset_hparams()
    source_hparams["bos_token"] = None
    source_hparams["data_name"] = "source"
    target_hparams = _default_mono_text_dataset_hparams()
    target_hparams.update(
        {
            "vocab_share": False,
            "embedding_init_share": False,
            "processing_share": False,
            "data_name": "target"
        }
    )
    return {
        "source_dataset": source_hparams,
        "target_dataset": target_hparams
    }

# pylint: disable=too-many-instance-attributes, too-many-public-methods
class PairedTextData(TextDataBase):
    """Text data processor that reads parallel source and target text.
    This can be used in, e.g., seq2seq models.

    Args:
        hparams (dict): Hyperparameters. See :meth:`default_hparams` for the
            defaults.

    By default, the processor reads raw data files, performs tokenization,
    batching and other pre-processing steps, and results in a TF Dataset
    whose element is a python `dict` including six fields:

        - "source_text":
            A string Tensor of shape `[batch_size, max_time]` containing
            the **raw** text toknes of source sequences. `max_time` is the
            length of the longest sequence in the batch.
            Short sequences in the batch are padded with **empty string**.
            By default only EOS token is appended to each sequence.
            Out-of-vocabulary tokens are **NOT** replaced with UNK.
        - "source_text_ids":
            An `int64` Tensor of shape `[batch_size, max_time]`
            containing the token indexes of source sequences.
        - "source_length":
            An `int` Tensor of shape `[batch_size]` containing the
            length of each source sequence in the batch (including BOS and/or
            EOS if added).
        - "target_text":
            A string Tensor as "source_text" but for target sequences. By
            default both BOS and EOS are added.
        - "target_text_ids":
            An `int64` Tensor as "source_text_ids" but for target sequences.
        - "target_length":
            An `int` Tensor of shape `[batch_size]` as "source_length" but for
            target sequences.

    If :attr:`'variable_utterance'` is set to `True` in :attr:`'source_dataset'`
    and/or :attr:`'target_dataset'` of :attr:`hparams`, the corresponding
    fields "source_*" and/or "target_*" are respectively changed to contain
    variable utterance text data, as in :class:`~texar.data.MonoTextData`.

    The above field names can be accessed through :attr:`source_text_name`,
    :attr:`source_text_id_name`, :attr:`source_length_name`,
    :attr:`source_utterance_cnt_name`, and those prefixed with `target_`,
    respectively.

    Example:

        .. code-block:: python

            hparams={
                'source_dataset': {'files': 's', 'vocab_file': 'vs'},
                'target_dataset': {'files': ['t1', 't2'], 'vocab_file': 'vt'},
                'batch_size': 1
            }
            data = PairedTextData(hparams)
            iterator = DataIterator(data)
            batch = iterator.get_next()

            iterator.switch_to_dataset(sess) # initializes the dataset
            batch_ = sess.run(batch)
            # batch_ == {
            #    'source_text': [['source', 'sequence', '<EOS>']],
            #    'source_text_ids': [[5, 10, 2]],
            #    'source_length': [3]
            #    'target_text': [['<BOS>', 'target', 'sequence', '1', '<EOS>']],
            #    'target_text_ids': [[1, 6, 10, 20, 2]],
            #    'target_length': [5]
            # }

    """
    def __init__(self, hparams):
        TextDataBase.__init__(self, hparams)
        with tf.name_scope(self.name, self.default_hparams()["name"]):
            self._make_data()

    @staticmethod
    def default_hparams():
        """Returns a dicitionary of default hyperparameters.

        .. code-block:: python

            {
                # (1) Hyperparams specific to text dataset
                "source_dataset": {
                    "files": [],
                    "compression_type": None,
                    "vocab_file": "",
                    "embedding_init": {},
                    "delimiter": " ",
                    "max_seq_length": None,
                    "length_filter_mode": "truncate",
                    "pad_to_max_seq_length": False,
                    "bos_token": None,
                    "eos_token": "<EOS>",
                    "other_transformations": [],
                    "variable_utterance": False,
                    "utterance_delimiter": "|||",
                    "max_utterance_cnt": 5,
                    "data_name": "source",
                },
                "target_dataset": {
                    # ...
                    # Same fields are allowed as in "source_dataset" with the
                    # same default values, except the
                    # following new fields/values:
                    "bos_token": "<BOS>"
                    "vocab_share": False,
                    "embedding_init_share": False,
                    "processing_share": False,
                    "data_name": "target"
                }
                # (2) General hyperparams
                "num_epochs": 1,
                "batch_size": 64,
                "allow_smaller_final_batch": True,
                "shuffle": True,
                "shuffle_buffer_size": None,
                "shard_and_shuffle": False,
                "num_parallel_calls": 1,
                "prefetch_buffer_size": 0,
                "max_dataset_size": -1,
                "seed": None,
                "name": "paired_text_data",
                # (3) Bucketing
                "bucket_boundaries": [],
                "bucket_batch_sizes": None,
                "bucket_length_fn": None,
            }

        Here:

        1. Hyperparameters in the :attr:`"source_dataset"` and
        attr:`"target_dataset"` fields have the same definition as those
        in :meth:`texar.data.MonoTextData.default_hparams`, for source and
        target text, respectively.

        For the new hyperparameters in "target_dataset":

            "vocab_share" : bool
                Whether to share the vocabulary of source.
                If `True`, the vocab file of target is ignored.

            "embedding_init_share" : bool
                Whether to share the embedding initial value of source. If
                `True`, :attr:`"embedding_init"` of target is ignored.

                :attr:`"vocab_share"` must be true to share the embedding
                initial value.

            "processing_share" : bool
                Whether to share the processing configurations of source,
                including
                "delimiter", "bos_token", "eos_token", and
                "other_transformations".

        2. For the **general** hyperparameters, see
        :meth:`texar.data.DataBase.default_hparams` for details.

        3. For **bucketing** hyperparameters, see
        :meth:`texar.data.MonoTextData.default_hparams` for details, except
        that the default bucket_length_fn is the maximum sequence length
        of source and target sequences.

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
        tgt_bos_token = utils.default_str(tgt_bos_token,
                                          SpecialTokens.BOS)
        tgt_eos_token = utils.default_str(tgt_eos_token,
                                          SpecialTokens.EOS)
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

    @staticmethod
    def _get_name_prefix(src_hparams, tgt_hparams):
        name_prefix = [
            src_hparams["data_name"], tgt_hparams["data_name"]]
        if name_prefix[0] == name_prefix[1]:
            raise ValueError("'data_name' of source and target "
                             "datasets cannot be the same.")
        return name_prefix

    @staticmethod
    def _make_processor(src_hparams, tgt_hparams, data_spec, name_prefix):
        # Create source data decoder
        data_spec_i = data_spec.get_ith_data_spec(0)
        src_decoder, src_trans, data_spec_i = MonoTextData._make_processor(
            src_hparams, data_spec_i, chained=False)
        data_spec.set_ith_data_spec(0, data_spec_i, 2)

        # Create target data decoder
        tgt_proc_hparams = tgt_hparams
        if tgt_hparams["processing_share"]:
            tgt_proc_hparams = copy.copy(src_hparams)
            try:
                tgt_proc_hparams["variable_utterance"] = \
                        tgt_hparams["variable_utterance"]
            except TypeError:
                tgt_proc_hparams.variable_utterance = \
                        tgt_hparams["variable_utterance"]
        data_spec_i = data_spec.get_ith_data_spec(1)
        tgt_decoder, tgt_trans, data_spec_i = MonoTextData._make_processor(
            tgt_proc_hparams, data_spec_i, chained=False)
        data_spec.set_ith_data_spec(1, data_spec_i, 2)

        tran_fn = dsutils.make_combined_transformation(
            [[src_decoder] + src_trans, [tgt_decoder] + tgt_trans],
            name_prefix=name_prefix)

        data_spec.add_spec(name_prefix=name_prefix)

        return tran_fn, data_spec

    @staticmethod
    def _make_length_filter(src_hparams, tgt_hparams,
                            src_length_name, tgt_length_name,
                            src_decoder, tgt_decoder):
        src_filter_fn = MonoTextData._make_length_filter(
            src_hparams, src_length_name, src_decoder)
        tgt_filter_fn = MonoTextData._make_length_filter(
            tgt_hparams, tgt_length_name, tgt_decoder)
        combined_filter_fn = dsutils._make_combined_filter_fn(
            [src_filter_fn, tgt_filter_fn])
        return combined_filter_fn

    def _process_dataset(self, dataset, hparams, data_spec):
        name_prefix = PairedTextData._get_name_prefix(
            hparams["source_dataset"], hparams["target_dataset"])
        tran_fn, data_spec = self._make_processor(
            hparams["source_dataset"], hparams["target_dataset"],
            data_spec, name_prefix=name_prefix)

        num_parallel_calls = hparams["num_parallel_calls"]
        dataset = dataset.map(
            lambda *args: tran_fn(dsutils.maybe_tuple(args)),
            num_parallel_calls=num_parallel_calls)

        # Filters by length
        src_length_name = dsutils._connect_name(
            data_spec.name_prefix[0],
            data_spec.decoder[0].length_tensor_name)
        tgt_length_name = dsutils._connect_name(
            data_spec.name_prefix[1],
            data_spec.decoder[1].length_tensor_name)
        filter_fn = self._make_length_filter(
            hparams["source_dataset"], hparams["target_dataset"],
            src_length_name, tgt_length_name,
            data_spec.decoder[0], data_spec.decoder[1])
        if filter_fn:
            dataset = dataset.filter(filter_fn)

        # Truncates data count
        dataset = dataset.take(hparams["max_dataset_size"])

        return dataset, data_spec

    def _make_bucket_length_fn(self):
        length_fn = self._hparams.bucket_length_fn
        if not length_fn:
            length_fn = lambda x: tf.maximum(
                x[self.source_length_name], x[self.target_length_name])
        elif not is_callable(length_fn):
            # pylint: disable=redefined-variable-type
            length_fn = utils.get_function(length_fn, ["texar.custom"])
        return length_fn

    def _make_padded_shapes(self, dataset, src_decoder, tgt_decoder):
        src_text_and_id_shapes = {}
        if self._hparams.source_dataset.pad_to_max_seq_length:
            src_text_and_id_shapes = \
                    MonoTextData._make_padded_text_and_id_shapes(
                        dataset, self._hparams.source_dataset, src_decoder,
                        self.source_text_name, self.source_text_id_name)

        tgt_text_and_id_shapes = {}
        if self._hparams.target_dataset.pad_to_max_seq_length:
            tgt_text_and_id_shapes = \
                    MonoTextData._make_padded_text_and_id_shapes(
                        dataset, self._hparams.target_dataset, tgt_decoder,
                        self.target_text_name, self.target_text_id_name)

        padded_shapes = dataset.output_shapes
        padded_shapes.update(src_text_and_id_shapes)
        padded_shapes.update(tgt_text_and_id_shapes)

        return padded_shapes

    def _make_data(self):
        self._src_vocab, self._tgt_vocab = self.make_vocab(
            self._hparams.source_dataset, self._hparams.target_dataset)

        tgt_hparams = self._hparams.target_dataset
        if not tgt_hparams.vocab_share and tgt_hparams.embedding_init_share:
            raise ValueError("embedding_init can be shared only when vocab "
                             "is shared. Got `vocab_share=False, "
                             "emb_init_share=True`.")
        self._src_embedding, self._tgt_embedding = self.make_embedding(
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

        # Processing.
        data_spec = dsutils._DataSpec(
            dataset=dataset, dataset_size=self._dataset_size,
            vocab=[self._src_vocab, self._tgt_vocab],
            embedding=[self._src_embedding, self._tgt_embedding])
        dataset, data_spec = self._process_dataset(
            dataset, self._hparams, data_spec)
        self._data_spec = data_spec
        self._decoder = data_spec.decoder
        self._src_decoder = data_spec.decoder[0]
        self._tgt_decoder = data_spec.decoder[1]

        # Batching
        length_fn = self._make_bucket_length_fn()
        padded_shapes = self._make_padded_shapes(
            dataset, self._src_decoder, self._tgt_decoder)
        dataset = self._make_batch(
            dataset, self._hparams, length_fn, padded_shapes)

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

        Note that this is the total data count in the raw files, before any
        filtering and truncation.
        """
        if not self._dataset_size:
            # pylint: disable=attribute-defined-outside-init
            self._dataset_size = count_file_lines(
                self._hparams.source_dataset.files)
        return self._dataset_size

    @property
    def vocab(self):
        """A pair instances of :class:`~texar.data.Vocab` that are source
        and target vocabs, respectively.
        """
        return self._src_vocab, self._tgt_vocab

    @property
    def source_vocab(self):
        """The source vocab, an instance of :class:`~texar.data.Vocab`.
        """
        return self._src_vocab

    @property
    def target_vocab(self):
        """The target vocab, an instance of :class:`~texar.data.Vocab`.
        """
        return self._tgt_vocab

    @property
    def source_embedding_init_value(self):
        """The `Tensor` containing the embedding value of source data
        loaded from file. `None` if embedding is not specified.
        """
        if self._src_embedding is None:
            return None
        return self._src_embedding.word_vecs

    @property
    def target_embedding_init_value(self):
        """The `Tensor` containing the embedding value of target data
        loaded from file. `None` if embedding is not specified.
        """
        if self._tgt_embedding is None:
            return None
        return self._tgt_embedding.word_vecs

    def embedding_init_value(self):
        """A pair of `Tensor` containing the embedding values of source and
        target data loaded from file.
        """
        src_emb = self.source_embedding_init_value
        tgt_emb = self.target_embedding_init_value
        return src_emb, tgt_emb

    @property
    def source_text_name(self):
        """The name of the source text tensor, "source_text" by default.
        """
        name = dsutils._connect_name(
            self._data_spec.name_prefix[0],
            self._src_decoder.text_tensor_name)
        return name

    @property
    def source_length_name(self):
        """The name of the source length tensor, "source_length" by default.
        """
        name = dsutils._connect_name(
            self._data_spec.name_prefix[0],
            self._src_decoder.length_tensor_name)
        return name

    @property
    def source_text_id_name(self):
        """The name of the source text index tensor, "source_text_ids" by
        default.
        """
        name = dsutils._connect_name(
            self._data_spec.name_prefix[0],
            self._src_decoder.text_id_tensor_name)
        return name

    @property
    def source_utterance_cnt_name(self):
        """The name of the source text utterance count tensor,
        "source_utterance_cnt" by default.
        """
        if not self._hparams.source_dataset.variable_utterance:
            raise ValueError(
                "`utterance_cnt_name` of source data is undefined.")
        name = dsutils._connect_name(
            self._data_spec.name_prefix[0],
            self._src_decoder.utterance_cnt_tensor_name)
        return name

    @property
    def target_text_name(self):
        """The name of the target text tensor, "target_text" bt default.
        """
        name = dsutils._connect_name(
            self._data_spec.name_prefix[1],
            self._tgt_decoder.text_tensor_name)
        return name

    @property
    def target_length_name(self):
        """The name of the target length tensor, "target_length" by default.
        """
        name = dsutils._connect_name(
            self._data_spec.name_prefix[1],
            self._tgt_decoder.length_tensor_name)
        return name

    @property
    def target_text_id_name(self):
        """The name of the target text index tensor, "target_text_ids" by
        default.
        """
        name = dsutils._connect_name(
            self._data_spec.name_prefix[1],
            self._tgt_decoder.text_id_tensor_name)
        return name

    @property
    def target_utterance_cnt_name(self):
        """The name of the target text utterance count tensor,
        "target_utterance_cnt" by default.
        """
        if not self._hparams.target_dataset.variable_utterance:
            raise ValueError(
                "`utterance_cnt_name` of target data is undefined.")
        name = dsutils._connect_name(
            self._data_spec.name_prefix[1],
            self._tgt_decoder.utterance_cnt_tensor_name)
        return name

    @property
    def text_name(self):
        """The name of text tensor, "text" by default.
        """
        return self._src_decoder.text_tensor_name

    @property
    def length_name(self):
        """The name of length tensor, "length" by default.
        """
        return self._src_decoder.length_tensor_name

    @property
    def text_id_name(self):
        """The name of text index tensor, "text_ids" by default.
        """
        return self._src_decoder.text_id_tensor_name

    @property
    def utterance_cnt_name(self):
        """The name of the text utterance count tensor, "utterance_cnt" by
        default.
        """
        if self._hparams.source_dataset.variable_utterance:
            return self._src_decoder.utterance_cnt_tensor_name
        if self._hparams.target_dataset.variable_utterance:
            return self._tgt_decoder.utterance_cnt_tensor_name
        raise ValueError("`utterance_cnt_name` is not defined.")
