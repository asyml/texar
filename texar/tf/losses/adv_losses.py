# Copyright 2018 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
    """Computes adversarial losses of real/fake binary discrimination game.

    .. role:: python(code)
       :language: python

    Args:
        real_data (Tensor or array): Real data of shape
            `[num_real_examples, ...]`.
        fake_data (Tensor or array): Fake data of shape
            `[num_fake_examples, ...]`. `num_real_examples` does not
            necessarily equal `num_fake_examples`.
        discriminator_fn: A callable takes data (e.g., :attr:`real_data` and
            :attr:`fake_data`) and returns the logits of being real. The
            signature of `discriminator_fn` must be:
            :python:`logits, ... = discriminator_fn(data)`.
            The return value of `discriminator_fn` can be the logits, or
            a tuple where the logits are the first element.

        mode (str): Mode of the generator loss. Either "max_real" or "min_fake".

            - **"max_real"** (default): minimizing the generator loss is to\
            maximize the probability of fake data being classified as real.

            - **"min_fake"**: minimizing the generator loss is to minimize the\
            probability of fake data being classified as fake.

    Returns:
        A tuple `(generator_loss, discriminator_loss)` each of which is
        a scalar Tensor, loss to be minimized.
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
