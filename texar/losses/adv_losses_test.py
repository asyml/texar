#
"""
Tests adversarial loss related functions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf

from texar.losses.adv_losses import binary_adversarial_losses

class AdvLossesTest(tf.test.TestCase):
    """Tests adversarial losses.
    """
    def test_binary_adversarial_losses(self):
        """Tests :meth:`~texar.losses.adv_losses.binary_adversarial_losse`.
        """
        batch_size = 16
        data_dim = 64
        real_data = tf.zeros([batch_size, data_dim], dtype=tf.float32)
        fake_data = tf.ones([batch_size, data_dim], dtype=tf.float32)
        const_logits = tf.zeros([batch_size], dtype=tf.float32)
        # Use a dumb discriminator that always outputs logits=0.
        gen_loss, disc_loss = binary_adversarial_losses(
            real_data, fake_data, lambda x: const_logits)
        gen_loss_2, disc_loss_2 = binary_adversarial_losses(
            real_data, fake_data, lambda x: const_logits, mode="min_fake")

        with self.test_session() as sess:
            gen_loss_, disc_loss_ = sess.run([gen_loss, disc_loss])
            gen_loss_2_, disc_loss_2_ = sess.run([gen_loss_2, disc_loss_2])
            self.assertAlmostEqual(gen_loss_, -gen_loss_2_)
            self.assertAlmostEqual(disc_loss_, disc_loss_2_)


if __name__ == "__main__":
    tf.test.main()
