"""Custom function for TSF."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

def len_pair(x):
    return tf.maximum(x["source_length"], x["target_length"])
