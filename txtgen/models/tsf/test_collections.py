
import pdb
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import utils

a = tf.Variable("a", [None])
utils.collect_named_outputs("collections", "a", a)

aa = tf.get_collection("collections")
pdb.set_trace()
d = utils.convert_collection_to_dict("collections")
