from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from txtgen.modules.encoders.rnn_encoders import ForwardRNNEncoder
from txtgen.core import optimization as opt
from txtgen.context import is_train

def adversarial_losses(true_data,
                       generated_data,
                       embedding=None,
                       vocab_size=None,
                       hparams=None,
                       ):
    discriminator = ForwardRNNEncoder(embedding=embedding, vocab_size=vocab_size, hparams=hparams)
    true_cost = discriminator_result(discriminator, true_data, 1, False)
    fake_cost = discriminator_result(discriminator, generated_data, 0, True)
    disc_cost = tf.reduce_sum(true_cost + fake_cost) #divide by batch size?
    disc_train_op, disc_global_step = opt.get_train_op(disc_cost)
    generate_cost = tf.reduce_sum(discriminator_result(discriminator, generated_data, 1, True))
    return disc_train_op, disc_global_step, generate_cost, disc_cost

def discriminator_result(discriminator, data, label, reuse=False):
    """Loss for both generated data and true data
    """
    _, state = discriminator(data)
    with tf.variable_scope('discriminator', reuse=reuse):
        logits = tf.layers.dense(state[0], 1)
        cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.ones_like(logits) * label)
        return cost

'''
integrate disc optimizer in the loss calculation? 

class AdversarialLosses():

    """Computes the adversarial loss
    Preliminary version
    Returns:
        A single average loss.
    """
    def __init__(self,
                 embedding=None,
                 vocab_size=None,
                 hparams=None,
                 ):
        self.true_data = tf.placeholder(tf.int32, name="true_data")
        self.generated_data = tf.placeholder(tf.int32, name="generated_data")
        self.true_data_sequence_length = tf.placeholder(tf.int32, name="true_data_sequence_length")
        self.generated_data_sequence_length = tf.placeholder(tf.int32, name="generated_data_sequence_length")
        self.discriminator = ForwardRNNEncoder(embedding=embedding, vocab_size=vocab_size, hparams=hparams)
        true_cost = self.discriminator_result(self.true_data, 1, False)
        self.generate_cost = self.discriminator_result(self.generated_data, 0, True)
        self.discriminator_cost = tf.reduce_sum(true_cost + self.generate_cost)
        self.disc_train_op, self.disc_global_step = opt.get_train_op(self.discriminator_cost)

    def discriminator_result(self, data, label, reuse=False):
        """Loss for both generated data and true data
        """
        _, state = self.discriminator(data)
        with tf.variable_scope('discriminator', reuse=reuse):
            logits = tf.layers.dense(state, 1)
            cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.ones_like(logits) * label)
            return cost

    def adversarial_losses(self,
                           true_data,
                           generated_data,
                           ):
        # pylint: disable=not-context-manager
        if is_train():
            with tf.Session() as sess:
                # train optimizer
                sess.run([self.disc_train_op, self.disc_global_step, self.discriminator_cost],
                    feed_dict={self.true_data: true_data, self.generated_data: generated_data})
        with tf.Session() as sess:
            generator_cost = sess.run([self.generate_cost], feed_dict={self.generated_data: generated_data})
        return generator_cost'''