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
        #true_inputs = tf.random_uniform([batch_size, max_time],
        #                           maxval=vocab_size,
        #                           dtype=tf.int32)
        true_inputs = np.zeros([batch_size, max_time], dtype="int32")
        #generate_inputs = tf.random_uniform([batch_size, max_time + 3],
        #                                maxval=vocab_size,
        #                                dtype=tf.int32)
        generate_inputs = np.ones([batch_size, max_time], dtype="int32")
        true_inputs_ph = tf.placeholder(tf.int32, [batch_size, max_time])
        generate_inputs_ph = tf.placeholder(tf.int32, [batch_size, max_time])
        discriminator = ForwardRNNEncoder(embedding=embedding, vocab_size=vocab_size, hparams=hparams)
        disc_train_op, disc_global_step, generator_loss, disc_loss = adversarial_losses(true_inputs_ph, generate_inputs_ph, discriminator)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            for i in range(10000):
                g_loss, d_loss, _, _ = sess.run([generator_loss, disc_loss, disc_global_step, disc_train_op], feed_dict={context.is_train(): True, true_inputs_ph: true_inputs, generate_inputs_ph: generate_inputs})
                print("generator_loss", g_loss)
                print("disc_loss", d_loss)
                #print("true inputs", true_inputs)

if __name__ == "__main__":
    tf.test.main()
