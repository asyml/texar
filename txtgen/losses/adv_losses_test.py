from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
import numpy as np
from txtgen import context


from .adv_losses import adversarial_losses

class AdvLossesTest(tf.test.TestCase):

    def test_adv_losses(self):
        vocab_size = 4
        max_time = 8
        batch_size = 16
        true_inputs = tf.random_uniform([batch_size, max_time],
                                   maxval=vocab_size,
                                   dtype=tf.int32)
        generate_inputs = tf.random_uniform([batch_size, max_time],
                                        maxval=vocab_size,
                                        dtype=tf.int32)
        _, disc_global_step, generator_loss, disc_loss = adversarial_losses(true_inputs, generate_inputs, vocab_size=vocab_size)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            for i in range(10000):
                g_loss, d_loss, _ = sess.run([generator_loss, disc_loss, disc_global_step], feed_dict={context.is_train(): True})
                print("generator_loss", g_loss)
                print("disc_loss", d_loss)

if __name__ == "__main__":
    tf.test.main()