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
            :attr:`fake_data`) and returns the logits of being real. The
            signature of :attr:`discriminator_fn` must be:

                `logits, ... = discriminator_fn(data)`

        mode (str): Mode of the generator loss. Either `max_real` or `min_fake`.

            If `max_real` (default), minimizing the generator loss is to
            maximize the probability of fake data being classified as real.

            If `min_fake`, minimizing the generator loss is to minimize the
            probability of fake data being classified as fake.

    Returns:
        (scalar Tensor, scalar Tensor): (generator_loss, discriminator_loss).
    """
    real_logits = discriminator_fn(real_data)
    if isinstance(real_logits, (list, tuple)):
        real_logits = real_logits[0]
    real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=real_logits, labels=tf.ones_like(real_logits)))

    fake_logits = discriminator_fn(fake_data)
    if isinstance(fake_logits, (list, tuple)):
        fake_logits = fake_logits[0]
    fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=fake_logits, labels=tf.zeros_like(fake_logits)))

    d_loss = real_loss + fake_loss

    if mode == "min_fake":
        g_loss = - fake_loss
    elif mode == "max_real":
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=fake_logits, labels=tf.ones_like(fake_logits)))
    else:
        raise ValueError("Unknown mode: %s. Only 'min_fake' and 'max_real' "
                         "are allowed.")

    return g_loss, d_loss
