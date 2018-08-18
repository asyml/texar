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
Various optimization related utilities.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import inspect

import tensorflow as tf

from texar.hyperparams import HParams
from texar.utils import utils

# pylint: disable=too-many-arguments, no-member

__all__ = [
    "default_optimization_hparams",
    "get_optimizer_fn",
    "get_learning_rate_decay_fn",
    "get_gradient_clip_fn",
    "get_train_op"
]

def default_optimization_hparams():
    """Returns a `dict` of default hyperparameters of training operator
    and their default values

    .. code-block:: python

        {
            "optimizer": {
                "type": "AdamOptimizer",
                "kwargs": {
                    "learning_rate": 0.001
                }
            },
            "learning_rate_decay": {
                "type": "",
                "kwargs": {},
                "min_learning_rate": 0.,
                "start_decay_step": 0,
                "end_decay_step": utils.MAX_SEQ_LENGTH,
            },
            "gradient_clip": {
                "type": "",
                "kwargs": {}
            },
            "gradient_noise_scale": None,
            "name": None
        }

    Here:

    "optimizer" :

    """
    return {
        "optimizer": {
            "type": "AdamOptimizer",
            "kwargs": {
                "learning_rate": 0.001
            }
        },
        "learning_rate_decay": {
            "type": "",
            "kwargs": {},
            "min_learning_rate": 0.,
            "start_decay_step": 0,
            "end_decay_step": utils.MAX_SEQ_LENGTH,
        },
        "gradient_clip": {
            "type": "",
            "kwargs": {}
        },
        "gradient_noise_scale": None,
        # TODO(zhiting): allow module-level control of gradient_multipliers
        "name": None
    }

def get_optimizer_fn(hparams=None):
    """Returns a function `optimizer_fn` of making optimizer instance, along
    with the optimizer class.

    .. role:: python(code)
       :language: python

    The function has the signiture:
        :python:`optimizer_fn(learning_rate=None) -> optimizer class instance`

    See the :attr:`"optimizer"` field of
    :meth:`~texar.core.optimization.default_optimization_hparams` for all
    hyperparameters and default values.

    The optimizer class must be a subclass of
    :tf_main:`tf.train.Optimizer <train/Optimizer>`.

    Args:
        hparams (dict or HParams, optional): hyperparameters. Missing
            hyperparameters are set to default values automatically.

    Returns:
        If "type" in :attr:`hparams` is a string or optimizer class, returns
        (`optimizer_fn`, optimizer class),

        If "type" in :attr:`hparams` is an optimizer instance, returns
        (the optimizer instance, optimizer class)
    """
    if hparams is None or isinstance(hparams, dict):
        hparams = HParams(
            hparams, default_optimization_hparams()["optimizer"])

    opt = hparams["type"]
    if isinstance(opt, tf.train.Optimizer):
        return opt, type(opt)
    else:
        opt_modules = ['tensorflow.train',
                       'tensorflow.contrib.opt',
                       'texar.custom']
        try:
            opt_class = utils.check_or_get_class(opt, opt_modules,
                                                 tf.train.Optimizer)
        except TypeError:
            raise ValueError(
                "Unrecognized optimizer. Must be string name of the "
                "optimizer class, or the class which is a subclass of "
                "tf.train.Optimizer, or an instance of the subclass of "
                "Optimizer.")

    def _get_opt(learning_rate=None):
        opt_kwargs = hparams["kwargs"].todict()
        fn_args = set(inspect.getargspec(opt_class.__init__).args)
        if 'learning_rate' in fn_args and learning_rate is not None:
            opt_kwargs["learning_rate"] = learning_rate
        return opt_class(**opt_kwargs)

    return _get_opt, opt_class

def get_learning_rate_decay_fn(hparams=None):
    """Creates learning rate decay function based on the hyperparameters.

    See the :attr:`learning_rate_decay` field in
    :meth:`~texar.core.optimization.default_optimization_hparams` for all
    hyperparameters and default values.

    Args:
        hparams (dict or HParams, optional): hyperparameters. Missing
            hyperparameters are set to default values automatically.

    Returns:
        function or None: If :attr:`hparams["type"]` is specified, returns a
        function that takes :attr:`learning_rate` and :attr:`global_step` and
        returns a scalar Tensor representing the decayed learning rate. If
        :attr:`hparams["type"]` is empty, returns `None`.
    """
    if hparams is None or isinstance(hparams, dict):
        hparams = HParams(
            hparams, default_optimization_hparams()["learning_rate_decay"])

    fn_type = hparams["type"]
    if fn_type is None or fn_type == "":
        return None

    fn_modules = ["tensorflow.train", "texar.custom"]
    decay_fn = utils.get_function(fn_type, fn_modules)
    fn_kwargs = hparams["kwargs"]
    if fn_kwargs is HParams:
        fn_kwargs = fn_kwargs.todict()

    start_step = tf.to_int32(hparams["start_decay_step"])
    end_step = tf.to_int32(hparams["end_decay_step"])

    def lr_decay_fn(learning_rate, global_step):
        """Learning rate decay function.

        Args:
            learning_rate (float or Tensor): The original learning rate.
            global_step (int or scalar int Tensor): optimization step counter.

        Returns:
            scalar float Tensor: decayed learning rate.
        """
        offset_global_step = tf.maximum(
            tf.minimum(tf.to_int32(global_step), end_step) - start_step, 0)
        if decay_fn == tf.train.piecewise_constant:
            decayed_lr = decay_fn(x=offset_global_step, **fn_kwargs)
        else:
            fn_kwargs_ = {
                "learning_rate": learning_rate,
                "global_step": offset_global_step}
            fn_kwargs_.update(fn_kwargs)
            decayed_lr = utils.call_function_with_redundant_kwargs(
                decay_fn, fn_kwargs_)

            decayed_lr = tf.maximum(decayed_lr, hparams["min_learning_rate"])

        return decayed_lr

    return lr_decay_fn


def get_gradient_clip_fn(hparams=None):
    """Creates a gradient clipping function based on the hyperparameters.

    See the :attr:`gradient_clip` field in
    :meth:`~texar.core.optimization.default_optimization_hparams` for all
    hyperparameters and default values.

    The gradient clipping function takes a list of `(gradients, variables)`
    tuples and returns a list of `(clipped_gradients, variables)` tuples.
    Typical examples include
    :tf_main:`tf.clip_by_global_norm <clip_by_global_norm>`,
    :tf_main:`tf.clip_by_value <clip_by_value>`,
    :tf_main:`tf.clip_by_norm <clip_by_norm>`,
    :tf_main:`tf.clip_by_average_norm <clip_by_average_norm>`, etc.

    Args:
        hparams (dict or HParams, optional): hyperparameters. Missing
            hyperparameters are set to default values automatically.

    Returns:
        function or `None`: If :attr:`hparams["type"]` is specified, returns
        the respective function. If :attr:`hparams["type"]` is empty,
        returns `None`.
    """
    if hparams is None or isinstance(hparams, dict):
        hparams = HParams(
            hparams, default_optimization_hparams()["gradient_clip"])
    fn_type = hparams["type"]
    if fn_type is None or fn_type == "":
        return None

    fn_modules = ["tensorflow", "texar.custom"]
    clip_fn = utils.get_function(fn_type, fn_modules)
    clip_fn_args = inspect.getargspec(clip_fn).args
    fn_kwargs = hparams["kwargs"]
    if isinstance(fn_kwargs, HParams):
        fn_kwargs = fn_kwargs.todict()

    def grad_clip_fn(grads_and_vars):
        """Gradient clipping function.

        Args:
            grads_and_vars (list): A list of `(gradients, variables)` tuples.

        Returns:
            list: A list of `(clipped_gradients, variables)` tuples.
        """
        grads, vars_ = zip(*grads_and_vars)
        if clip_fn == tf.clip_by_global_norm:
            clipped_grads, _ = clip_fn(t_list=grads, **fn_kwargs)
        elif 't_list' in clip_fn_args:
            clipped_grads = clip_fn(t_list=grads, **fn_kwargs)
        elif 't' in clip_fn_args:     # e.g., tf.clip_by_value
            clipped_grads = [clip_fn(t=grad, **fn_kwargs) for grad in grads]

        return list(zip(clipped_grads, vars_))

    return grad_clip_fn


def get_train_op(loss, variables=None, learning_rate=None,
                 global_step=None, increment_global_step=True, hparams=None):
    """Creates a training op.

    Args:
        loss: A scalar Tensor representing the loss to optimize.
        variables (optional): A list of Variables to optimize. If
            `None`, all trainable variables are used.
        learning_rate (float or Tensor, optional): If `None`, learning rate
            specified in :attr:`hparams`, or the default learning rate
            of the optimizer will be used (if exists).
        global_step (optional): A scalar int Tensor. Step counter to update on
            each step unless :attr:`increment_global_step` is `False`.
            Learning rate decay requires requires :attr:`global_step`.
        increment_global_step (bool): Whether to increment
            :attr:`global_step`. This is useful if the :attr:`global_step` is
            used in multiple training ops per training step (e.g. to optimize
            different parts of the model) to avoid incrementing
            :attr:`global_step` more times than necessary.
        hparams (dict or HParams, optional): hyperparameters. Missing
            hyperparameters are set to default values automatically. See
            :meth:`~texar.core.optimization.default_optimization_hparams` for
            all hyperparameters and default values.

    Returns:
        tuple: (train_op, global_step). If :attr:`global_step` is provided, the
        same :attr:`global_step` variable is returned, otherwise a new global
        step is created and returned.
    """
    hparams = HParams(hparams, default_optimization_hparams())

    opt_hparams = hparams["optimizer"]
    optimizer_fn, optimizer_class = get_optimizer_fn(opt_hparams)

    if learning_rate is None:
        learning_rate = opt_hparams["kwargs"].get("learning_rate", None)
    if learning_rate is None:
        # Try to get learning_rate from the default value of the
        # optimizer's argument
        opt_argspec = utils.get_default_arg_values(optimizer_class.__init__)
        learning_rate = opt_argspec.get("learning_rate", None)

    grad_clip_fn = get_gradient_clip_fn(hparams["gradient_clip"])

    lr_decay_fn = get_learning_rate_decay_fn(hparams["learning_rate_decay"])

    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=global_step,
        learning_rate=learning_rate,
        optimizer=optimizer_fn,
        gradient_noise_scale=hparams["gradient_noise_scale"],
        clip_gradients=grad_clip_fn,
        learning_rate_decay_fn=lr_decay_fn,
        variables=variables,
        name=hparams["name"],
        increment_global_step=increment_global_step)

    return train_op
