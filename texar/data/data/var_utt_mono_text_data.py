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

from texar.data.data import data_utils
from texar.data.data.mono_text_data import MonoTextData
from texar.data.data_decoders import VarUttTextDataDecoder

# pylint: disable=invalid-name, arguments-differ, not-context-manager

__all__ = [
    "VarUttMonoTextData"
]

# pylint: disable=no-member

class VarUttMonoTextData(MonoTextData):
    """Mono text data where each data sample includes variable number of
    utterances.

    This is for the use of, e.g., dialog systems, where we want to encode
    utterances in dialog history.

    Args:
        hparams (dict): Hyperparameters. See :meth:`default_hparams` for the
            defaults.
    """

    def __init__(self, hparams):
        MonoTextData.__init__(self, hparams)
        with tf.name_scope(self.name, self.default_hparams()["name"]):
            self._make_data()

    @staticmethod
    def default_hparams():
        """Returns a dicitionary of default hyperparameters.
        """
        #TODO(zhiting): add docs
        hparams = MonoTextData.default_hparams()
        hparams["dataset"]["max_utterance_cnt"] = 5
        hparams["name"] = "var_utt_mono_text_data"
        return hparams

    @staticmethod
    def _make_processor(dataset_hparams, data_spec):
        # Create data decoder
        decoder = VarUttTextDataDecoder(
            delimiter=dataset_hparams["delimiter"],
            bos_token=dataset_hparams["bos_token"],
            eos_token=dataset_hparams["eos_token"],
            max_seq_length=dataset_hparams["max_seq_length"],
            max_utterance_cnt=dataset_hparams["max_utterance_cnt"],
            token_to_id_map=data_spec.vocab.token_to_id_map)

        # Create other transformations
        data_spec.add_spec(decoder=decoder)
        other_trans = MonoTextData._make_other_transformations(
            dataset_hparams["other_transformations"], data_spec)

        # Process data
        chained_tran = data_utils.make_chained_transformation(
            [decoder] + other_trans)

        return chained_tran, data_spec

    @property
    def utterance_cnt_name(self):
        """The name of utterance count tensor.
        """
        return self._decoder.utterance_cnt_tensor_name

