from texar.core import layers
import tensorflow as tf
def attention_bias_lower_triangle(length):
    """Create an bias tensor to be added to attention logits.
    Allows a query to attend to all positions up to and including its own.
    Args:
     length: a Scalar.
    Returns:
        a `Tensor` with shape [1, 1, length, length].
    """
    return attention_bias_local(length, -1, 0)

def attention_bias_local(length, max_backward, max_forward):
    """Create an bias tensor to be added to attention logits.
    A position may attend to positions at most max_distance from it,
    forward and backwards.
    This does not actually save any computation.
    Args:
        length: int
        max_backward: int, maximum distance backward to attend. Negative values
            indicate unlimited.
        max_forward: int, maximum distance forward to attend. Negative values
            indicate unlimited.
    Returns:
        a `Tensor` with shape [1, 1, length, length].
        [batch_size, num_heads, queri_len, queri_len]
    """
    band = layers.ones_matrix_band_part(
            length,
            length,
            max_backward,
            max_forward,
            out_shape=[1, 1, length, length])
    return -1e9 * (1.0 - band)

def attention_bias_ignore_padding(memory_padding):
    """Create an bias tensor to be added to attention logits.
    Args:
        memory_padding: a float `Tensor` with shape [batch, memory_length].
    Returns:
        a `Tensor` with shape [batch, 1, 1, memory_length].
        each dim corresponding to batch_size, num_heads, queries_len, memory_length
    """
    ret = memory_padding * -1e9
    return tf.expand_dims(tf.expand_dims(ret, axis=1), axis=1)
