"""
This file gives some help files for model training.
"""
import tensorflow as tf
import numpy as np
import random

def set_random_seed(myseed):
    tf.set_random_seed(myseed)
    np.random.seed(myseed)
    random.seed(myseed)

global max_src_in_batch, max_tgt_in_batch

def batch_size_fn(new, count, sofar):
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch, max_tgt_in_batch = 0, 0
    max_src_in_batch = max(max_src_in_batch, len(new[0] + 1))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new[1] + 1))
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)

def adjust_lr(fstep, opt_config):
    if opt_config['learning_rate_schedule'] == 'static':
        lr = opt_config['static_lr']
    else:
        lr = opt_config['lr_constant'] \
            * tf.minimum(1.0, (fstep / opt_config['warmup_steps'])) \
            * tf.rsqrt(tf.maximum(fstep, opt_config['warmup_steps']))
    return lr

