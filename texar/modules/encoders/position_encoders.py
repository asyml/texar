import math
import abc
import six

import tensorflow as tf



def make_positions(sequence_length, maximum_length=None):
  """Builds a sequence of positions.
  The first position is 1 as the 0 index is reserved to padding positions.
  Args:
    sequence_length: The length of each sequence as a ``tf.Tensor`` of shape
      :math:`[B]`.
    maximum_length: Optional size of the returned time dimension. Otherwise it
      is the maximum of :obj:`sequence_length`.
  Returns:
    The sequence of positions as a ``tf.Tensor`` of shape :math:`[B, T]`.
  """
  if maximum_length is None:
    maximum_length = tf.reduce_max(sequence_length)

  batch_size = tf.shape(sequence_length)[0]

  # Make 0 the position of padding.
  position = tf.range(maximum_length) + 1
  position = tf.tile(position, [batch_size])
  position = tf.reshape(position, [batch_size, -1])

  mask = tf.sequence_mask(
      sequence_length, maxlen=maximum_length, dtype=position.dtype)

  position = position * mask

  return position


@six.add_metaclass(abc.ABCMeta)
class PositionEncoder(object):
  """Base class for position encoders."""

  def __init__(self):
      pass
  def __call__(self, inputs, sequence_length=None):
    """Shortcut for `apply`."""
    return self.apply(inputs, sequence_length=sequence_length)

  def apply(self, inputs, sequence_length=None):
    """Apply position encoding to inputs.
    Args:
      inputs: The inputs of shape :math:`[B, T, D]`.
      sequence_length: The length of each sequence of shape :math:`[B]`.
        If ``None``, sequences are assumed to have the same length.
    Returns:
      A ``tf.Tensor`` of shape :math:`[B, T, D]` where :math:`D` depends on the
      :attr:`reducer`.
    """
    timesteps = tf.shape(inputs)[1]

    if sequence_length is None:
      batch_size = tf.shape(inputs)[0]
      sequence_length = tf.fill([batch_size], timesteps)

    input_dim = inputs.get_shape().as_list()[-1]

    with tf.variable_scope("position_encoding"):
      position_encoding = self.encode_sequence(
          sequence_length,
          input_dim,
          maximum_length=timesteps,
          dtype=inputs.dtype)
      return tf.add_n([inputs, position_encoding])

  def apply_one(self, inputs, position):
    """Apply position encoding to one input.
    This is usually used during dynamic decoding.
    Args:
      inputs: The inputs of shape :math:`[B, 1, D]`.
      position: The position to encode.
    Returns:
      A ``tf.Tensor`` of shape :math:`[B, 1, D]` where :math:`D` depends on the
      :attr:`reducer`.
    """
    batch_size = tf.shape(inputs)[0]
    input_dim = inputs.get_shape().as_list()[-1]

    position = tf.tile([position], [batch_size])
    position = tf.expand_dims(position, 1)

    with tf.variable_scope("position_encoding"):
      position_encoding = self.encode(position, input_dim, dtype=inputs.dtype)
      return tf.add_n([inputs, position_encoding])

  @abc.abstractmethod
  def encode(self, positions, depth, dtype=tf.float32):
    """Creates position encodings.
    Args:
      position: The positions to encode of shape :math:`[B, ...]`.
      depth: The encoding depth :math:`D`.
      dtype: The encoding type.
    Returns:
      A ``tf.Tensor`` of shape :math:`[B, ..., D]`.
    """
    raise NotImplementedError()

  def encode_sequence(self,
                      sequence_length,
                      depth,
                      maximum_length=None,
                      dtype=tf.float32):
    """Creates position encodings for sequences.
    Args:
      sequence_length: The length of each sequence of shape :math:`[B]`.
      depth: The encoding depth :math:`D`.
      maximum_length: Optional size of the returned time dimension. Otherwise
        it is the maximum of :obj:`sequence_length`.
      dtype: The encoding type.
    Returns:
      A ``tf.Tensor`` of shape :math:`[B, T, D]`.
    """
    positions = make_positions(sequence_length, maximum_length=maximum_length)
    return self.encode(positions, depth, dtype=dtype)


class PositionEmbedder(PositionEncoder):
  """Encodes position with a lookup table."""

  def __init__(self, maximum_position=128):
    """Initializes the position encoder.
    Args:
      maximum_position: The maximum position to embed. Positions greater
        than this value will be set to :obj:`maximum_position`.
    """
    super(PositionEmbedder, self).__init__()
    self.maximum_position = maximum_position

  def encode(self, positions, depth, dtype=tf.float32):
    positions = tf.minimum(positions, self.maximum_position)
    embeddings = tf.get_variable(
        "w_embs", shape=[self.maximum_position + 1, depth], dtype=dtype)
    return tf.nn.embedding_lookup(embeddings, positions)


class SinusoidalPositionEncoder(PositionEncoder):
  """Encodes positions with sine waves as described in
  https://arxiv.org/abs/1706.03762.
  """

  def encode(self, positions, depth, dtype=tf.float32):
    batch_size = tf.shape(positions)[0]
    positions = tf.cast(positions, dtype)

    log_timescale_increment = math.log(10000) / (depth / 2 - 1)
    inv_timescales = tf.exp(tf.range(depth / 2, dtype=dtype) * -log_timescale_increment)
    inv_timescales = tf.reshape(tf.tile(inv_timescales, [batch_size]), [batch_size, -1])
    scaled_time = tf.expand_dims(positions, -1) * tf.expand_dims(inv_timescales, 1)

    return tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=2)

