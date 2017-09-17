# -*- coding: utf-8 -*-
# author: Tiancheng Zhao
import tensorflow as tf


def sample_gaussian(mu, logvar):
    """
    Sample a sample from a multivariate Gaussian distribution with a diagonal covariance matrix using the 
    reparametrization trick.
    
    TODO: this should be better be a instance method in a Gaussian class.
    
    :param mu: a tensor of size [batch_size, variable_dim]. Batch_size can be None to support dynamic batching
    :param logvar: a tensor of size [batch_size, variable_dim]. Batch_size can be None.
    :return: 
    """
    epsilon = tf.random_normal(tf.shape(logvar), name="epsilon")
    std = tf.exp(0.5 * logvar)
    z = mu + tf.multiply(std, epsilon)
    return z
