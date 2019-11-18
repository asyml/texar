# Copyright 2019 The Texar Authors. All Rights Reserved.
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

import tensorflow as tf

__all__ = [
    'valid_modes',
    'is_train_mode',
    'is_eval_mode',
    'is_predict_mode',
]


def valid_modes():
    r"""Returns a set of possible values of mode.
    """
    return {tf.estimator.ModeKeys.TRAIN,
            tf.estimator.ModeKeys.EVAL,
            tf.estimator.ModeKeys.PREDICT}


def is_train_mode(mode, default=True):
    r"""Returns a python boolean indicating whether the mode is TRAIN.

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
    if mode not in valid_modes():
        raise ValueError('Unknown mode: {}'.format(mode))
    return mode == tf.estimator.ModeKeys.TRAIN


def is_eval_mode(mode, default=False):
    r"""Returns a python boolean indicating whether the mode is EVAL.

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
    if mode not in valid_modes():
        raise ValueError('Unknown mode: {}'.format(mode))
    return mode == tf.estimator.ModeKeys.EVAL


def is_predict_mode(mode, default=False):
    r"""Returns a python boolean indicating whether the mode is PREDICT.

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
    if mode not in valid_modes():
        raise ValueError('Unknown mode: {}'.format(mode))
    return mode == tf.estimator.ModeKeys.PREDICT
