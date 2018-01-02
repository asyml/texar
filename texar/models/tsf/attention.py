""" Implementations of attention layers.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals


from tensorflow.python.eager import context

import tensorflow as tf

def att_sum_bahdanau(v_att, keys, query):
  """Calculates a batch- and timweise dot product with a variable"""
  return tf.reduce_sum(v_att * tf.tanh(keys + tf.expand_dims(query, 1)), [2])

def att_sum_dot(keys, query):
  """Calculates a batch- and timweise dot product"""
  return tf.reduce_sum(keys * tf.expand_dims(query, 1), [2])

class AttentionLayer(tf.layers.Layer):
  def __init__(self, num_units, trainable=True, name=None, reuse=None,
               **kwargs):
    super(AttentionLayer, self).__init__(trainable=trainable, name=name,
                                         _reuse=reuse, **kwargs)
    self._num_units = num_units

  def build(self, _):
    pass

  def __call__(self, query, keys, values, values_length, scope=None):
    if scope is not None:
      with tf.variable_scope(scope,
                             custom_getter=self._get_variable) as scope:
        return super(AttentionLayer, self).__call__(
          query, keys, values, values_length, scope=scope)
    else:
      with tf.variable_scope(tf.get_variable_scope(),
                             custom_getter=self._get_variable):
        return super(AttentionLayer, self).__call__(
          query, keys, values, values_length)

  def _get_variable(self, getter, *args, **kwargs):
    variable = getter(*args, **kwargs)
    if context.in_graph_mode():
      trainable = (variable in tf.trainable_variables() or
                   (isinstance(variable, tf.PartitionedVariable) and
                    list(variable)[0] in tf.trainable_variables()))
    else:
      trainable = variable._trainable  # pylint: disable=protected-access
    if trainable and variable not in self._trainable_weights:
      self._trainable_weights.append(variable)
    elif not trainable and variable not in self._non_trainable_weights:
      self._non_trainable_weights.append(variable)
    return variable

  def score_fn(self, keys, query):
    raise NotImplementedError

  def call(self, query, keys, values, values_length):
    values_depth = values.get_shape().as_list()[-1]

    att_keys = tf.contrib.layers.fully_connected(
      inputs=keys,
      num_outputs=self._num_units,
      activation_fn=None,
      scope="att_keys")
    att_query = tf.contrib.layers.fully_connected(
      inputs=query,
      num_outputs=self._num_units,
      activation_fn=None,
      scope="att_query")

    scores = self.score_fn(att_keys, att_query)

    num_scores = tf.shape(scores)[1]
    scores_mask = tf.sequence_mask(
      lengths=tf.to_int32(values_length),
      maxlen=tf.to_int32(num_scores),
      dtype=tf.float32) 
    scores = scores * scores_mask + ((1.0 - scores_mask) * tf.float32.min)

    scores_normalized = tf.nn.softmax(scores, name="scores_normalized")

    context = tf.expand_dims(scores_normalized, 2) * values
    context = tf.reduce_sum(context, 1, name="context")
    context.set_shape([None, values_depth])

    return (scores_normalized, context)


class AttentionLayerBahdanau(AttentionLayer):
  def score_fn(self, keys, query):
    v_att = tf.get_variable(
      "v_att", shape=self._num_units, dtype=tf.float32)
    return att_sum_bahdanau(v_att, keys, query)

class AttentionLayerDot(AttentionLayer):
  def score_fn(self, keys, query):
    return att_sum_dot(keys, query)
