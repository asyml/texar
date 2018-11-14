import tensorflow as tf
import pprint

from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
_vars = tf.train.list_variables('cased_L-12_H-768_A-12/bert_model.ckpt')
pprint.pprint(_vars)
#print_tensors_in_checkpoint_file(latest_ckp, all_tensors=True, tensor_name='')
