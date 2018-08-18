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
    """Returns a `dict` of default hyperparameters of training op
    and their default values

    .. role:: python(code)
       :language: python

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
                "end_decay_step": inf
            },
            "gradient_clip": {
                "type": "",
                "kwargs": {}
            },
            "gradient_noise_scale": None,
            "name": None
        }

    Here:

    "optimizer" : dict
        Hyperparameters of a :tf_main:`tf.train.Optimizer <train/Optimizer>`.

        - **"type"** specifies the optimizer class. This can be

            - The string name or full module path of an optimizer class. \
            If the class name is provided, the class must be in module \
            :tf_main:`tf.train <train>`, \
            :tf_main:`tf.contrib.opt <contrib/opt>` or :mod:`texar.custom`.
            - An optimizer class.
            - An instance of an optimizer class.

            For example

            .. code-block:: python

                "type": "AdamOptimizer" # class name
                "type": "my_module.MyOptimizer" # module path
                "type": tf.contrib.opt.AdamWOptimizer # class
                "type": my_module.MyOptimizer # class
                "type": GradientDescentOptimizer(learning_rate=0.1) # instance
                "type": MyOptimizer(...) # instance

        - **"kwargs"** is a `dict` specifying keyword arguments for creating \
        the optimizer class instance, with :python:`opt_class(**kwargs)`. \
        Ignored if "type" is a class instance.

    "learning_rate_decay" : dict
        Hyperparameters of learning rate decay function. The learning rate
        starts decay from :attr:`"start_decay_step"` and keeps unchanged after
        :attr:`"end_decay_step"` or reaching :attr:`"min_learning_rate"`.

        The decay function is specified in "type" and "kwargs".

            - "type" can be a decay function or its name or module path. If \
            function name is provided, it must be from module \
            :tf_main:`tf.train <train>` or :mod:`texar.custom`.

            - "kwargs" is a `dict` of keyword arguments for the function \
            excluding arguments named "global_step" and "learning_rate".

        The function is called with
        :python:`lr = decay_fn(learning_rate=lr, global_step=offset_step,
        **kwargs)`, where `offset_step` is the global step offset as above.
        The only exception is :tf_main:`tf.train.piecewise_constant
        <train/piecewise_constant>` which is called with
        :python:`lr = piecewise_constant(x=offset_step, **kwargs)`.

    "gradient_clip" : dict
        Hyperparameters of gradient clipping. The gradient clipping function
        takes a list of `(gradients, variables)` tuples and returns a list
        of `(clipped_gradients, variables)` tuples. Typical examples include
        :tf_main:`tf.clip_by_global_norm <clip_by_global_norm>`,
        :tf_main:`tf.clip_by_value <clip_by_value>`,
        :tf_main:`tf.clip_by_norm <clip_by_norm>`,
        :tf_main:`tf.clip_by_average_norm <clip_by_average_norm>`, etc.

        "type" specifies the gradient clip function, and can be a function,
        or its name or mudule path. If function name is provided, the
        function must be from module :tf_main:`tf < >` or :mod:`texar.custom`.

        "kwargs" specifies keyword arguments to the function, except arguments
        named "t" or "t_list".

        The function is called with
        :python:`clipped_grads(, _) = clip_fn(t_list=grads, **kwargs)`
        (e.g., for :tf_main:`tf.clip_by_global_norm <clip_by_global_norm>`) or
        :python:`clipped_grads = [clip_fn(t=grad, **kwargs) for grad in grads]`
        (e.g., for :tf_main:`tf.clip_by_value <clip_by_value>`).

    "gradient_noise_scale" : float, optional
        Adds 0-mean normal noise scaled by this value to gradient.
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

    The function has the signiture
    :python:`optimizer_fn(learning_rate=None) -> optimizer class instance`

    See the :attr:`"optimizer"` field of
    :meth:`~texar.core.default_optimization_hparams` for all
    hyperparameters and default values.

    The optimizer class must be a subclass of
    :tf_main:`tf.train.Optimizer <train/Optimizer>`.

    Args:
        hparams (dict or HParams, optional): hyperparameters. Missing
            hyperparameters are set to default values automatically.

    Returns:
        - If hparams["type"] is a string or optimizer class, returns\
        `(optimizer_fn, optimizer class)`,

        - If hparams["type"] is an optimizer instance, returns \
        `(the optimizer instance, optimizer class)`
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
    :meth:`~texar.core.default_optimization_hparams` for all
    hyperparameters and default values.

    Args:
        hparams (dict or HParams, optional): hyperparameters. Missing
            hyperparameters are set to default values automatically.

    Returns:
        function or None: If hparams["type"] is specified, returns a
        function that takes `(learning_rate, step, **kwargs)` and
        returns a decayed learning rate. If
        hparams["type"] is empty, returns `None`.
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
    :meth:`~texar.core.default_optimization_hparams` for all
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
        function or `None`: If hparams["type"] is specified, returns
        the respective function. If hparams["type"] is empty,
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

    This is a wrapper of :tf_main:`tf.contrib.layers.optimize_loss
    <contrib/layers/optimize_loss>`.

    Args:
        loss: A scalar Tensor representing the loss to minimize.
        variables (optional): A list of Variables to optimize. If
            `None`, all trainable variables are used.
        learning_rate (float or Tensor, optional): If `None`, learning rate
            specified in :attr:`hparams`, or the default learning rate
            of the optimizer will be used (if exists).
        global_step (optional): A scalar int Tensor. Step counter to update on
            each step unless :attr:`increment_global_step` is `False`.
            Learning rate decay uses :attr:`global_step`.
            If `None`, it will be fetched from the default graph (see
            :tf_main:`tf.train.get_global_step <train/get_global_step>` for
            more details). If it has not been created, no step will be
            incremented with each weight update.
        increment_global_step (bool): Whether to increment
            :attr:`global_step`. This is useful if the :attr:`global_step` is
            used in multiple training ops per training step (e.g. to optimize
            different parts of the model) to avoid incrementing
            :attr:`global_step` more times than necessary.
        hparams (dict or HParams, optional): hyperparameters. Missing
            hyperparameters are set to default values automatically. See
            :func:`~texar.core.default_optimization_hparams` for
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
