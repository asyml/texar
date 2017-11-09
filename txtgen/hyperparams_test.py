"""
Unit tests for embedding related operations.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import pickle

import tempfile
import tensorflow as tf

from txtgen.hyperparams import HParams

# pylint: disable=no-member

class HParamsTest(tf.test.TestCase):
    """Tests hyperparameter related operations.
    """

    def test_hparams(self):
        """Tests the HParams class.
        """
        default_hparams = {
            "str": "str",
            "list": ['item1', 'item2'],
            "dict": {
                "key1": "value1",
                "key2": "value2"
            },
            "nested_dict": {
                "dict_l2": {
                    "key1_l2": "value1_l2"
                }
            },
            "kwargs": {
                "arg1": "argv1"
            },
        }
        hparams = {
            "dict": {"key1": "new_value"},
            "kwargs": {"arg2": "argv2"}
        }

        hparams_ = HParams(hparams, default_hparams)

        # Test HParams construction
        self.assertEqual(hparams_.str, default_hparams["str"])
        self.assertEqual(hparams_.list, default_hparams["list"])
        self.assertEqual(hparams_.dict.key1, hparams["dict"]["key1"])
        self.assertEqual(hparams_.kwargs.arg2, hparams["kwargs"]["arg2"])
        self.assertEqual(hparams_.nested_dict.dict_l2.key1_l2,
                         default_hparams["nested_dict"]["dict_l2"]["key1_l2"])

        self.assertEqual(len(hparams_), len(default_hparams))

        new_hparams = copy.deepcopy(default_hparams)
        new_hparams["dict"]["key1"] = hparams["dict"]["key1"]
        new_hparams["kwargs"].update(hparams["kwargs"])
        self.assertEqual(hparams_.todict(), new_hparams)

        self.assertTrue("dict" in hparams_)

        self.assertIsNone(hparams_.get('not_existed_name', None))
        self.assertEqual(hparams_.get('str'), default_hparams['str'])

        # Test HParams update related operations
        hparams_.str = "new_str"
        hparams_.dict = {"key3": "value3"}
        self.assertEqual(hparams_.str, "new_str")
        self.assertEqual(hparams_.dict.key3, "value3")

        hparams_.add_hparam("added_str", "added_str")
        hparams_.add_hparam("added_dict", {"key4": "value4"})
        hparams_.kwargs.add_hparam("added_arg", "added_argv")
        self.assertEqual(hparams_.added_str, "added_str")
        self.assertEqual(hparams_.added_dict.todict(), {"key4": "value4"})
        self.assertEqual(hparams_.kwargs.added_arg, "added_argv")

        # Test HParams I/O
        hparams_file = tempfile.NamedTemporaryFile()
        pickle.dump(hparams_, hparams_file)
        with open(hparams_file.name, 'rb') as hparams_file:
            hparams_loaded = pickle.load(hparams_file)
        self.assertEqual(hparams_loaded.todict(), hparams_.todict())


if __name__ == "__main__":
    tf.test.main()
