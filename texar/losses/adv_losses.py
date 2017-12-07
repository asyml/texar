#
"""
Adversarial losses.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def binary_adversarial_losses(real_data,
                              fake_data,
                              discriminator_fn,
                              mode="max_real"):
    """Computes adversarial loss of the real/fake binary classification game.

    Args:
        real_data (Tensor or array): Real data of shape
            `[num_real_examples, ...]`.
        fake_data (Tensor or array): Fake data of shape
            `[num_fake_examples, ...]`. `num_real_examples` does not necessarily
            equal `num_fake_examples`.
        discriminator_fn: A callable takes data (e.g., :attr:`real_data` and
            :attr:`fake_data`) and returns the logits of being real.
        mode (str): Mode of the generator loss. Either `max_real` or `min_fake`.

            If `max_real` (default), minimizing the generator loss is to
            maximize the probability of fake data being classified as real.

            If `min_fake`, minimizing the generator loss is to minimize the
            probability of fake data being classified as fake.

    Returns:
        (scalar Tensor, scalar Tensor): (generator_loss, discriminator_loss).
    """
    real_logits = discriminator_fn(real_data)
    real_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=real_logits, labels=tf.ones_like(real_logits))
    num_real_data = tf.shape(real_loss)[0]
    ave_real_loss = tf.reduce_sum(real_loss) / tf.to_float(num_real_data)
    fake_logits = discriminator_fn(fake_data)
    fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=fake_logits, labels=tf.zeros_like(fake_logits))
    num_fake_data = tf.shape(fake_loss)[0]
    ave_fake_loss = tf.reduce_sum(fake_loss) / tf.to_float(num_fake_data)
    disc_loss = ave_real_loss + ave_fake_loss
    if mode == "min_fake":
        gen_loss = - ave_fake_loss
    elif mode == "max_real":
        fake_loss_ = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=fake_logits, labels=tf.ones_like(fake_logits))
        gen_loss = tf.reduce_sum(fake_loss_) / tf.to_float(num_fake_data)
    else:
        raise ValueError("Unknown mode: %s. Only 'min_fake' and 'max_real' "
                         "are allowed.")
    return gen_loss, disc_loss

