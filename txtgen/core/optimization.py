#
"""
Various optimization related utilities.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

from txtgen.hyperparams import HParams
from txtgen.core import utils


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
            "min_learning_rate": None,
            "start_decay_step": 0,
            "end_decay_step": utils.MAX_SEQ_LENGTH,
        },
        "clip_gradients": {
            "type": "",
            "kwargs": {}
        }
    }

# TODO(zhiting): add YellowFin optimizer
def get_optimizer(hparams):
    """Creates an optimizer based on hyperparameters.

    See the :attr:"optimizer" field in
    :meth:`~txtgen.core.optimization.default_optimization_hparams` for the
    hyperparameters.

    Args:
        hparams (dict or HParams): hyperparameters.

    Returns:
        An instance of :class:`~tensorflow.train.Optimizer`.
    """
    opt_type = hparams["type"]
    opt_kwargs = hparams["kwargs"]
    if opt_kwargs is HParams:
        opt_kwargs = opt_kwargs.todict()
    opt_modules = ['txtgen.custom',
                   'tensorflow.train',
                   'tensorflow.contrib.opt']
    opt = utils.get_instance(opt_type, opt_kwargs, opt_modules)

    return opt

def get_learning_rate_decay_fn(hparams):
    """Creates learning rate decay function based on the hyperparameters.

    See the :attr:"learning_rate_decay" field in
    :meth:`~txtgen.core.optimization.default_optimization_hparams` for the
    hyperparameters.

    Args:
        hparams (dict or HParams): hyperparameters.

    Returns:
        function: A function that takes :attr:`learning_rate` and
        :attr:`global_step`, and returns a scalar Tensor representing the
        learning rate.
    """
    fn_type = hparams["type"]
    fn_modules = ["txtgen.custom", "tensorflow.train"]
    tf_decay_fn = utils.get_function(fn_type, fn_modules)
    fn_kwargs = hparams["kwargs"]
    if fn_kwargs is HParams:
        fn_kwargs = fn_kwargs.todict()

    # TODO
    def lr_decay_fn(learning_rate, global_step):
        offset_global_step = \
            tf.minimum(tf.to_int32(global_step),
                       tf.to_int32(hparams["end_decay_step"])) - \
            tf.to_int32(hparams["start_decay_step"])
        if tf_decay_fn == tf.train.piecewise_constant:
            pass
        else:
            decayed_lr = tf_decay_fn(learning_rate,
                                     global_step, )

