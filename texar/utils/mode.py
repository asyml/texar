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
Utility functions related to mode.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

from texar import context

__all__ = [
    "maybe_global_mode",
    "is_train_mode",
    "is_eval_mode",
    "is_predict_mode",
    "is_train_mode_py",
    "is_eval_mode_py",
    "is_predict_mode_py",
    "switch_dropout"
]

def maybe_global_mode(mode):
    """Returns :func:`texar.global_mode` if :attr:`mode` is `None`,
    otherwise returns :attr:`mode` as-is.
    """
    if mode is None:
        return context.global_mode()
    else:
        return mode

def is_train_mode(mode):
    """Returns a bool Tensor indicating whether the global mode is TRAIN.
    If :attr:`mode` is `None`, the mode is determined by
    :func:`texar.global_mode`.
    """
    if mode is None:
        return context.global_mode_train()
    else:
        return tf.equal(mode, tf.estimator.ModeKeys.TRAIN)

def is_eval_mode(mode):
    """Returns a bool Tensor indicating whether the global mode is EVAL.
    If :attr:`mode` is `None`, the mode is determined by
    :func:`texar.global_mode`.
    """
    if mode is None:
        return context.global_mode_eval()
    else:
        return tf.equal(mode, tf.estimator.ModeKeys.EVAL)

def is_predict_mode(mode):
    """Returns a bool Tensor indicating whether the global mode is PREDICT.
    If :attr:`mode` is `None`, the mode is determined by
    :func:`texar.global_mode`.
    """
    if mode is None:
        return context.global_mode_predict()
    else:
        return tf.equal(mode, tf.estimator.ModeKeys.PREDICT)

def is_train_mode_py(mode, default=True):
    """Returns a python boolean indicating whether the mode is TRAIN.

    Args:
        mode: A string taking value in
            :tf_main:`tf.estimator.ModeKeys <estimator/ModeKeys>`.
            Can be `None`.
        default (bool): The return value when :attr:`mode` is `None`. Default
            is `True`.

    Returns:
        A python boolean.
    """
    if mode is None:
        return default
    if mode not in context.valid_modes():
        raise ValueError('Unknown mode: {}'.format(mode))
    return mode == tf.estimator.ModeKeys.TRAIN

def is_eval_mode_py(mode, default=False):
    """Returns a python boolean indicating whether the mode is EVAL.

    Args:
        mode: A string taking value in
            :tf_main:`tf.estimator.ModeKeys <estimator/ModeKeys>`.
            Can be `None`.
        default (bool): The return value when :attr:`mode` is `None`. Default
            is `False`.

    Returns:
        A python boolean.
    """
    if mode is None:
        return default
    if mode not in context.valid_modes():
        raise ValueError('Unknown mode: {}'.format(mode))
    return mode == tf.estimator.ModeKeys.EVAL

def is_predict_mode_py(mode, default=False):
    """Returns a python boolean indicating whether the mode is PREDICT.

    Args:
        mode: A string taking value in
            :tf_main:`tf.estimator.ModeKeys <estimator/ModeKeys>`.
            Can be `None`.
        default (bool): The return value when :attr:`mode` is `None`. Default
            is `False`.

    Returns:
        A python boolean.
    """
    if mode is None:
        return default
    if mode not in context.valid_modes():
        raise ValueError('Unknown mode: {}'.format(mode))
    return mode == tf.estimator.ModeKeys.PREDICT

def switch_dropout(dropout_keep_prob, mode=None):
    """Turns off dropout when not in training mode.

    Args:
        dropout_keep_prob: Dropout keep probability in training mode
        mode (optional): A Tensor taking values of
            :tf_main:`tf.estimator.ModeKeys <estimator/ModeKeys>`.
            Dropout is activated if :attr:`mode` is `TRAIN`.
            If `None`, the mode is inferred from
            :func:`texar.global_mode`.

    Returns:
        A unit Tensor that equals the dropout keep probability in `TRAIN` mode,
        and `1.0` in other modes.
    """
    return 1. - (1. - dropout_keep_prob) * tf.to_float(is_train_mode(mode))
