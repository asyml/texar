import tensorflow as tf


def l2_loss(qvalue, action_input, y_input, **kwargs):
    temp = qvalue * action_input
    temp = tf.reduce_sum(input_tensor=temp, axis=1)
    loss = tf.reduce_sum((temp - y_input) ** 2.0)
    return loss
