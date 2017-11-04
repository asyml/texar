import tensorflow as tf
import math

def sinusoid_positional_encoder(self, inputs, num_units, zero_pad, scale):
    """
    Args:
        inputs:[batch_size, max_time, dim]
    Returns:
        positional embedding:[batch_size, max_time, dim]
    """
    max_time = tf.shape(inputs)[1]
    dim = tf.shape(inputs)[2]
    batch_size = tf.shape(inputs)[0]
    input_one = tf.tile(tf.expand_dims(tf.range(max_time), 0), batch_size, 1)
    position_block = tf.tile(tf.expand_dims(tf.range(max_time), 1), [1, dim // 2])
    unit_block = tf.tile(tf.expand_dims(tf.range(dim // 2), 0), [max_time, 1])
    rad_block = tf.pow(tf.div(position_block, tf.multiply(10000, 1)), tf.div(unit_block, dim // 2))
    sin_block = tf.sin(tf.cast(rad_block, tf.float32))
    cos_block = tf.cos(tf.cast(rad_block, tf.float32))
    lookup_table = tf.concat([sin_block, cos_block], axis = 1)
    if zero_pad:
        lookup_table = tf.concat((tf.zeros(shape = [1, dim]),lookup_table[1:, :]), 0)
    outputs = tf.nn.embedding_lookup(lookup_table, input_one)
    if scale:
        outputs = outputs * math.sqrt(num_units)
    return outputs
