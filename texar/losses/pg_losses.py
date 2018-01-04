"""
Loss functions for Policy Gradient
"""

import tensorflow as tf


def pg_loss(outputs, action_inputs, advantages, **kwargs):
    """
    Policy Gradient Loss Function
    :param outputs:
    :param action_inputs:
    :param advantages:
    :param kwargs:
    :return:
    """
    neg_log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=outputs, labels=action_inputs)
    return tf.reduce_mean(neg_log_probs * advantages)
