#
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

__all__ = [
    "default_optimization_hparams",
    "get_optimizer",
    "get_learning_rate_decay_fn",
    "get_gradient_clip_fn",
    "get_train_op"
]

def default_optimization_hparams():
    """Returns default hyperparameters of optimization.

    Returns:
        dict: A dictionary with the following structure and values:

    .. code-block:: python

        {
        }

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

# TODO(zhiting): add YellowFin optimizer
def get_optimizer(hparams=None):
    """Creates an optimizer based on hyperparameters.

    See the :attr:"optimizer" field in
    :meth:`~texar.core.optimization.default_optimization_hparams` for all
    hyperparameters and default values.

    Args:
        hparams (dict or HParams, optional): hyperparameters. Missing
            hyperparameters are set to default values automatically.

    Returns:
        An instance of :class:`~tensorflow.train.Optimizer`.
    """
    if hparams is None or isinstance(hparams, dict):
        hparams = HParams(
            hparams, default_optimization_hparams()["optimizer"])

    opt_type = hparams["type"]
    opt_kwargs = hparams["kwargs"].todict()
    opt_modules = ['texar.custom',
                   'tensorflow.train',
                   'tensorflow.contrib.opt']
    opt = utils.get_instance(opt_type, opt_kwargs, opt_modules)

    return opt

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

    fn_modules = ["texar.custom", "tensorflow.train"]
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
        offset_global_step = tf.minimum(
            tf.to_int32(global_step), end_step) - start_step
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
    if fn_kwargs is HParams:
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


def get_train_op(loss, variables=None, global_step=None,
                 increment_global_step=True, hparams=None):
    """Creates a training op.

    Args:
        loss (scalar Tensor): loss to optimize over.
        variables (list of Variables, optional): Variables to optimize. If
            `None`, all trainable variables are used.
        global_step (scalar int Tensor, optional): step counter to update on
            each step unless :attr:`increment_global_step` is `False`. If
            `None`, a new global step variable will be created.
        incremental_global_step (bool): Whether to increment
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

    if variables is None:
        variables = tf.trainable_variables()
    if global_step is None:
        global_step_name = None
        if hparams["name"] is not None:
            global_step_name = '_'.join([hparams["name"], 'step'])
        global_step = tf.Variable(0, name=global_step_name, trainable=False)

    optimizer = get_optimizer(hparams["optimizer"])

    learning_rate = hparams["optimizer"]["kwargs"].get("learning_rate", None)
    if learning_rate is None:
        # Try to get learning_rate from the default value of the
        # optimizer's argument
        opt_argspec = utils.get_default_arg_values(optimizer.__init__)
        if 'learning_rate' not in opt_argspec:
            raise ValueError(
                "`learning_rate` must be specified in "
                "hparams['optimizer']['kwargs'], if the optimizer does not "
                "have default value for it.")
        learning_rate = opt_argspec["learning_rate"]

    grad_clip_fn = get_gradient_clip_fn(hparams["gradient_clip"])

    lr_decay_fn = get_learning_rate_decay_fn(hparams["learning_rate_decay"])

    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=global_step,
        learning_rate=learning_rate,
        optimizer=optimizer,
        gradient_noise_scale=hparams["gradient_noise_scale"],
        clip_gradients=grad_clip_fn,
        learning_rate_decay_fn=lr_decay_fn,
        variables=variables,
        name=hparams["name"],
        increment_global_step=increment_global_step)

    return train_op, global_step
