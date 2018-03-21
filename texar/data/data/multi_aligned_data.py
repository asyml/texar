#
"""
Paired text data that consists of source text and target text.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf

from texar.hyperparams import HParams
from texar.core import utils
from texar.data.data.mono_text_data import _default_mono_text_dataset_hparams
from texar.data.data.text_data_base import TextDataBase
from texar.data.data.mono_text_data import MonoTextData
from texar.data.data import data_utils
from texar.data.vocabulary import Vocab
from texar.data.embedding import Embedding
from texar.data.constants import BOS_TOKEN, EOS_TOKEN

# pylint: disable=invalid-name, arguments-differ, not-context-manager
# pylint: disable=protected-access

__all__ = [
    "_default_dataset_hparams",
    "MultiAlignedTextData"
]

class DataTypes: # pylint: disable=old-style-class, no-init, too-few-public-methods
    """Enumeration of data types.
    """
    TEXT = "text"
    MULTI_TEXT = "multi_text"
    INT = "int"
    FLOAT = "float"

def _default_dataset_hparams():
    """Returns hyperparameters of a dataset with default values.
    """
    # TODO(zhiting): add more docs
    hparams = _default_mono_text_dataset_hparams()
    hparams.update({
        "data_type": DataTypes.TEXT,
        "data_name_prefix": None,
        "vocab_share_with": None,
        "embedding_init_share_with": None,
        "processing_share_with": None,
    })
    return hparams

# pylint: disable=too-many-instance-attributes
class MultiAlignedTextData(TextDataBase):
    """Data consists of multiple aligned parts.

    Args:
        hparams (dict): Hyperparameters. See :meth:`default_hparams` for the
            defaults.
    """
    def __init__(self, hparams):
        TextDataBase.__init__(self, hparams)
        # Defaultizes hparams of each dataset
        datasets_hparams = self._hparams.datasets
        defaultized_datasets_hparams = []
        for ds_hpms in datasets_hparams:
            defaultized_ds_hpms = HParams(ds_hpms, _default_dataset_hparams())
            defaultized_datasets_hparams.append(defaultized_ds_hpms)
        self._hparams.datasets = defaultized_datasets_hparams

        with tf.name_scope(self.name, self.default_hparams()["name"]):
            self._make_data()

    @staticmethod
    def default_hparams():
        """Returns a dicitionary of default hyperparameters.
        """
        hparams = TextDataBase.default_hparams()
        hparams["name"] = "multi_aligned_data"
        hparams["datasets"] = [_default_dataset_hparams()]
        return hparams

    @staticmethod
    def _raise_sharing_error(err_data, shr_data, hparam_name):
        raise ValueError(
            "Must only share specifications with a preceding dataset. "
            "Dataset %d has '%s=%d'" % (err_data, hparam_name, shr_data))

    @staticmethod
    def make_vocab(hparams):
        """Makes a list of vocabs based on the hparams.

        Args:
            hparams (list): A list of dataset hyperparameters.

        Returns:
            A list of :class:`texar.data.Vocab` instances. Some instances
            may be the same objects if they are set to be shared and have
            the same other configs.
        """
        if not isinstance(hparams, (list, tuple)):
            hparams = [hparams]

        vocabs = []
        for i, hparams_i in enumerate(hparams):
            if hparams_i["data_type"] != DataTypes.TEXT and \
                    hparams_i != DataTypes.MULTI_TEXT:
                vocabs.append(None)
                continue

            proc_shr = hparams_i["processing_share_with"]
            if proc_shr:
                bos_token = hparams[proc_shr]["bos_token"]
                eos_token = hparams[proc_shr]["eos_token"]
            else:
                bos_token = hparams_i["bos_token"]
                eos_token = hparams_i["eos_token"]
            bos_token = utils.default_string(bos_token, BOS_TOKEN)
            eos_token = utils.default_string(eos_token, EOS_TOKEN)

            vocab_shr = hparams_i["vocab_share_with"]
            if vocab_shr:
                if vocab_shr >= i:
                    MultiAlignedTextData._raise_sharing_error(
                        i, vocab_shr, "vocab_share_with")
                if not vocabs[vocab_shr]:
                    raise ValueError("Cannot share vocab with dataset %d which "
                                     "does not have a vocab." % vocab_shr)
                if bos_token == vocabs[vocab_shr].bos_token and \
                        eos_token == vocabs[vocab_shr].eos_token:
                    vocab = vocabs[vocab_shr]
                else:
                    vocab = Vocab(hparams[vocab_shr]["vocab_file"],
                                  bos_token=bos_token,
                                  eos_token=eos_token)
            else:
                vocab = Vocab(hparams_i["vocab_file"],
                              bos_token=bos_token,
                              eos_token=eos_token)
            vocabs.append(vocab)

        return vocabs

    @staticmethod
    def make_embedding(hparams, vocabs):
        """Optionally loads embeddings from files (if provided), and
        returns respective :class:`texar.data.Embedding` instances.
        """
        if not isinstance(hparams, (list, tuple)):
            hparams = [hparams]

        embs = []
        for i, hparams_i in enumerate(hparams):
            if hparams_i["data_type"] != DataTypes.TEXT and \
                    hparams_i != DataTypes.MULTI_TEXT:
                embs.append(None)
                continue

            emb_shr = hparams_i["embedding_init_share_with"]
            if emb_shr:
                if emb_shr >= i:
                    MultiAlignedTextData._raise_sharing_error(
                        i, emb_shr, "embedding_init_share_with")
                if not embs[emb_shr]:
                    raise ValueError("Cannot share embedding with dataset %d "
                                     "which does not have an embedding." %
                                     emb_shr)
                if emb_shr != hparams_i["vocab_share_with"]:
                    raise ValueError("'embedding_init_share_with' != "
                                     "vocab_share_with. embedding_init can "
                                     "be shared only when vocab is shared.")
                emb = embs[emb_shr]
            else:
                emb = None
                emb_file = hparams_i["embedding_init"]["file"]
                if emb_file and emb_file != "":
                    emb = Embedding(vocabs[i].token_to_id_map_py,
                                    hparams_i["embedding_init"])
            embs.append(emb)

        return embs

    def _make_dataset(self):
        datasets = []
        for _, hparams_i in enumerate(self._hparams.datasets):
            if hparams_i.data_type in {DataTypes.TEXT, DataTypes.MULTI_TEXT,
                                       DataTypes.INT, DataTypes.FLOAT}:
                dataset = tf.data.TextLineDataset(
                    hparams_i.files,
                    compression_type=hparams_i.compression_type)
                datasets.append(dataset)
            else:
                raise ValueError("Unknown data type: %s" % hparams_i.data_type)
        return tf.data.Dataset.zip(tuple(datasets))

    @staticmethod
    def _make_processor(dataset_hparams, data_spec, name_prefix=None):
        processors = []
        for i, hparams_i in enumerate(dataset_hparams):
            data_spec_i = data_spec.get_ith_data_spec(i)

            data_type = hparams_i["data_type"]
            if data_type == DataTypes.TEXT:
                processor, data_spec_i = MonoTextData._make_processor(
                    hparams_i, data_spec_i)
            else:
                raise ValueError("Unsupported data type: %s" % data_type)

            processors.append(processor)
            data_spec.set_ith_data_spec(i, data_spec_i, len(dataset_hparams))

        if not name_prefix:
            name_prefix = [str(i) for i in range(len(dataset_hparams))]
        tran_fn = data_utils.make_combined_transformation(
            processors, name_prefix=name_prefix)

        data_spec.add_spec(name_prefix=name_prefix)

        return tran_fn, data_spec

    def _process_dataset(self, dataset, hparams, data_spec):
        tran_fn, data_spec = self.MultiAlignedTextData._make_processor(
            hparams["datasets"], data_spec, hparams["data_name_prefix"])
        num_parallel_calls = hparams["num_parallel_calls"]
        dataset = dataset.map(
            lambda *args: tran_fn(data_utils.maybe_tuple(args)),
            num_parallel_calls=num_parallel_calls)
        return dataset, data_spec

    def _make_length_fn(self):
        length_fn = self._hparams.bucket_length_fn
        if not length_fn:
            # Uses the length of the first text data
            i = -1
            for i, hparams_i in range(self._hparams.datasets):
                if hparams_i["data_type"] == DataTypes.TEXT:
                    break
            if i < 0:
                raise ValueError("Undefined `length_fn`.")
            length_fn = lambda x: x[self.length_name(i)]
        elif not utils.is_callable(length_fn):
            # pylint: disable=redefined-variable-type
            length_fn = utils.get_function(length_fn, ["texar.custom"])
        return length_fn

    def _make_data(self):
        self._vocab = self.make_vocab(self._hparams.datasets)
        self._embedding = self.make_embedding(self._hparams.datasets,
                                              self._vocab)

        # Create dataset
        dataset = self._make_dataset()
        dataset, dataset_size = self._shuffle_dataset(
            dataset, self._hparams, self._hparams.datasets[0].files)
        self._dataset_size = dataset_size

        # Processing
        data_spec = data_utils._DataSpec(dataset=dataset,
                                         dataset_size=self._dataset_size,
                                         vocab=self._vocab,
                                         embedding=self._embedding)
        dataset, data_spec = self._process_dataset(
            dataset, self._hparams, data_spec)
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
                self._hparams.datasets[0].files)
        return self._dataset_size

    def vocab(self, i):
        """Returns the :class:`~texar.data.Vocab` of the :attr:`i`-th dataset.
        `None` if the :attr:`i`-th dataaset is not of text type.
        """
        return self._vocab[i]

    def embedding_init_value(self, i):
        """Returns the `Tensor` of embedding init value of the :attr:`i`-th
        dataset. `None` if the :attr:`i`-th dataaset is not of text type.
        """
        return self._embedding[i]

    def text_name(self, i):
        """The name of text tensor of the :attr:`i`-th dataset. If the
        :attr:`i`-th dataaset is not of text type, the result is un-defined.
        """
        if not self._data_spec.decoder[i]:
            raise ValueError("text name of datset %d undefined." % i)
        name = "{}_{}".format(self._data_spec.prefix_name[i],
                              self._data_spec.decoder[i].text_tensor_name)
        return name

    def length_name(self, i):
        """The name of length tensor of the :attr:`i`-th dataset. If the
        :attr:`i`-th dataaset is not of text type, the result is un-defined.
        """
        if not self._data_spec.decoder[i]:
            raise ValueError("length name of datset %d undefined." % i)
        name = "{}_{}".format(self._data_spec.prefix_name[i],
                              self._data_spec.decoder[i].length_tensor_name)
        return name

    def text_id_name(self, i):
        """The name of length tensor of the :attr:`i`-th dataset. If the
        :attr:`i`-th dataaset is not of text type, the result is un-defined.
        """
        if not self._data_spec.decoder[i]:
            raise ValueError("text id name of datset %d undefined." % i)
        name = "{}_{}".format(self._data_spec.prefix_name[i],
                              self._data_spec.decoder[i].text_id_tensor_name)
        return name

