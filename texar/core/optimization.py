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

import re
import tensorflow as tf

from texar.hyperparams import HParams
from texar.utils import utils

# pylint: disable=too-many-arguments, no-member

__all__ = [
    "default_optimization_hparams",
    "get_optimizer_fn",
    "get_learning_rate_decay_fn",
    "get_gradient_clip_fn",
    "get_optimizer",
    "get_train_op",
    "AdamWeightDecayOptimizer",
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
            :tf_main:`tf.contrib.opt <contrib/opt>` or :mod:`texar.custom` \
            , :mod:`texar.core.optimization`
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
            :tf_main:`tf.train <train>` or :mod:`texar.custom`, \
            :mod:`texar.core.optimization`.

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
        function must be from module :tf_main:`tf < >` or :mod:`texar.custom`,
        :mod:`texar.core.optimization`.


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
    opt_modules = ['tensorflow.train',
                   'tensorflow.contrib.opt',
                   'texar.core.optimization',
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
        fn_args = set(utils.get_args(opt_class.__init__))
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
    clip_fn_args = utils.get_args(clip_fn)
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

def _get_static_lr(learning_rate=None, optimizer_class=None, hparams=None):
    """Return the base static learning_rate.
        A helper function for creating the optimization function.
    """
    hparams = HParams(hparams, default_optimization_hparams())
    opt_hparams = hparams['optimizer']
    if learning_rate is None:
        learning_rate = opt_hparams["kwargs"].get("learning_rate", None)
    if learning_rate is None:
        # Try to get learning_rate from the default value of the
        # optimizer's argument
        opt_argspec = utils.get_default_arg_values(optimizer_class.__init__)
        learning_rate = opt_argspec.get("learning_rate", None)
    return learning_rate

def get_optimizer(learning_rate=None, global_step=None, hparams=None):

    """Creates a optimizer instance.

    Args:
        learning_rate (float or Tensor, optional): If `None`, learning rate
            specified in :attr:`hparams`, or the default learning rate
            of the optimizer (if exists) is used.
        global_step (optional): A scalar int Tensor. Step counter to update on
            each step unless :attr:`increment_global_step` is `False`.
            Learning rate decay uses :attr:`global_step`.
            If `None`, it will be fetched from the default graph (see
            :tf_main:`tf.train.get_global_step <train/get_global_step>` for
            more details). If it has not been created, no step will be
            incremented with each weight update.
        hparams (dict or HParams, optional): hyperparameters. Missing
            hyperparameters are set to default values automatically. See
            :func:`~texar.core.default_optimization_hparams` for
            all hyperparameters and default values.

    Returns:
        optimizer: the tf.train.Optimizer instance specified in hparams.
    """
    hparams = HParams(hparams, default_optimization_hparams())

    opt_hparams = hparams["optimizer"]
    optimizer_fn, optimizer_class = get_optimizer_fn(opt_hparams)

    static_lr = _get_static_lr(learning_rate, optimizer_class, hparams)

    lr_decay_fn = get_learning_rate_decay_fn(hparams["learning_rate_decay"])
    if lr_decay_fn is not None:
        learning_rate = lr_decay_fn(learning_rate=static_lr,
                                    global_step=global_step)
    else:
        learning_rate = static_lr

    tf.summary.scalar("learning_rate", learning_rate)

    optimizer = optimizer_fn(learning_rate=learning_rate)

    return optimizer

def get_train_op(loss, variables=None,
                 optimizer=None, learning_rate=None,
                 global_step=None, increment_global_step=True, hparams=None):
    """Creates a training op.

    This is a wrapper of :tf_main:`tf.contrib.layers.optimize_loss
    <contrib/layers/optimize_loss>`.

    Args:
        loss: A scalar Tensor representing the loss to minimize.
        variables (optional): A list of Variables to optimize. If
            `None`, all trainable variables are used.
        optimizer (optional): An tf.train.Optimizer instance. If `None`,
            use the setting in `hparams` to create the optimizer.
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
        train_op: the operator used for variables optimization.
    """
    hparams = HParams(hparams, default_optimization_hparams())
    grad_clip_fn = get_gradient_clip_fn(hparams["gradient_clip"])

    if not isinstance(optimizer, tf.train.Optimizer):
        opt_hparams = hparams["optimizer"]
        optimizer_fn, optimizer_class = get_optimizer_fn(opt_hparams)
        learning_rate = _get_static_lr(learning_rate, optimizer_class, hparams)
        lr_decay_fn = get_learning_rate_decay_fn(
            hparams["learning_rate_decay"])
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

    else:
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=global_step,
            learning_rate=None,
            optimizer=optimizer,
            gradient_noise_scale=hparams["gradient_noise_scale"],
            clip_gradients=grad_clip_fn,
            variables=variables,
            name=hparams["name"],
            increment_global_step=increment_global_step)

    return train_op

class AdamWeightDecayOptimizer(tf.train.Optimizer):
    """
    A basic Adam optimizer that includes "correct" L2 weight decay.
    Copied from the google BERT repo.
    Except that in `apply_gradient` function, we add the support to increment
    the passed global step parameter, to make it more compatible to
    tf.train.Optimizer implementation.
    """

    def __init__(self,
                 learning_rate,
                 weight_decay_rate=0.0,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-6,
                 exclude_from_weight_decay=None,
                 name="AdamWeightDecayOptimizer"):
        """Constructs a AdamWeightDecayOptimizer."""
        super(AdamWeightDecayOptimizer, self).__init__(False, name)

        self.learning_rate = learning_rate
        self.weight_decay_rate = weight_decay_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.exclude_from_weight_decay = exclude_from_weight_decay

    # pylint: disable=too-many-locals
    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """See base class."""
        with tf.name_scope(name, self._name) as name:
            assignments = []
            for (grad, param) in grads_and_vars:
                if grad is None or param is None:
                    continue

                param_name = self._get_variable_name(param.name)

                m = tf.get_variable(
                    name=param_name + "/adam_m",
                    shape=param.shape.as_list(),
                    dtype=tf.float32,
                    trainable=False,
                    initializer=tf.zeros_initializer())
                v = tf.get_variable(
                    name=param_name + "/adam_v",
                    shape=param.shape.as_list(),
                    dtype=tf.float32,
                    trainable=False,
                    initializer=tf.zeros_initializer())

                # Standard Adam update.
                next_m = (tf.multiply(self.beta_1, m)\
                          + tf.multiply(1.0 - self.beta_1,
                                        grad))
                next_v = (tf.multiply(self.beta_2, v)\
                          + tf.multiply(1.0 - self.beta_2,
                                        tf.square(grad)))

                update = next_m / (tf.sqrt(next_v) + self.epsilon)

                # Just adding the square of the weights to the loss function is
                # *not* the correct way of using L2 regularization/weight decay
                # with Adam, since that will interact with the m and v
                # parameters in strange ways.
                # Instead we want ot decay the weights in a manner that doesn't
                # interact with the m/v parameters.
                # This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                if self._do_use_weight_decay(param_name):
                    update += self.weight_decay_rate * param

                update_with_lr = self.learning_rate * update

                next_param = param - update_with_lr

                assignments.extend(
                    [param.assign(next_param),
                     m.assign(next_m),
                     v.assign(next_v)])

            update_ops = assignments
            if global_step is None:
                apply_updates = self._finish(update_ops, name)
            else:
                with tf.control_dependencies([self._finish(update_ops,
                                                           "update")]):
                    with tf.colocate_with(global_step):
                        apply_updates = tf.assign_add(global_step, 1, name=name)

        return apply_updates

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay_rate:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name
