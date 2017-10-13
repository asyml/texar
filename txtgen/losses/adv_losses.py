from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from txtgen.modules.encoders.rnn_encoders import ForwardRNNEncoder
from txtgen.core import optimization as opt
from txtgen.context import is_train

def adversarial_losses(true_data,
                       generated_data,
                       discriminator
                       ):
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

