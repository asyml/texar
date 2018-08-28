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
This file gives some help files for model training.
"""
import tensorflow as tf
import numpy as np
import random
import math

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
            * min(1.0, (fstep / opt_config['warmup_steps'])) \
            * (1 / math.sqrt(max(fstep, opt_config['warmup_steps'])))
    return lr

