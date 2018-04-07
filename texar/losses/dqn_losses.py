"""
Loss functions for DQN
"""

import tensorflow as tf


def l2_loss(qvalue, action_input, y_input, **kwargs):
    """
    L2 loss function.

    Args:
         qvalue(Tensor):
         action_input(Tensor):
         y_input(Tensor):

    """
    temp = qvalue * action_input
    temp = tf.reduce_sum(input_tensor=temp, axis=1)
    loss = tf.reduce_sum((temp - y_input) ** 2.0)
    return loss
