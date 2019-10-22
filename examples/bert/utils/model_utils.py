"""
Model utility functions
"""

import tensorflow as tf


def get_lr(global_step, num_train_steps, num_warmup_steps, static_lr):
    """
    Calculate the learinng rate given global step and warmup steps.
    The learinng rate is following a linear warmup and linear decay.
    """
    learning_rate = tf.constant(value=static_lr,
                                shape=[], dtype=tf.float32)

    learning_rate = tf.train.polynomial_decay(
        learning_rate,
        global_step,
        num_train_steps,
        end_learning_rate=0.0,
        power=1.0,
        cycle=False)

    if num_warmup_steps:
        global_steps_int = tf.cast(global_step, tf.int32)
        warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

        global_steps_float = tf.cast(global_steps_int, tf.float32)
        warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

        warmup_percent_done = global_steps_float / warmup_steps_float
        warmup_learning_rate = static_lr * warmup_percent_done

        is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
        learning_rate = ((1.0 - is_warmup) * learning_rate
                         + is_warmup * warmup_learning_rate)

    return learning_rate
