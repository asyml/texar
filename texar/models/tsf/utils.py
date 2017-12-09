"""
Utils for tsf.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.layers.python.layers import utils

from texar.hyperparams import HParams

def register_collection(collection_name, tensors):
  for n, t in tensors:
    utils.collect_named_outputs(collection_name, n, t)

  return utils.convert_collection_to_dict(collection_name)


def filter_hparams(hparams, prefix):
  hparams = {k[len(prefix)+1:]: v
             for k, v in hparams if k.startswith(prefix)}
  return HParams(None, hparams)
