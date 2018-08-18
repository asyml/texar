#
"""
Unit tests for various optimization related utilities.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import tensorflow as tf

import texar.core.optimization as opt
from texar.utils import utils


class OptimizationTest(tf.test.TestCase):
    """Tests optimization.
    """

    def test_get_optimizer(self):
        """Tests get_optimizer.
        """
        default_optimizer_fn, optimizer_class = opt.get_optimizer_fn(
            opt.default_optimization_hparams()["optimizer"])
        default_optimizer = default_optimizer_fn(1.0)
        self.assertTrue(optimizer_class, tf.train.Optimizer)
        self.assertIsInstance(default_optimizer, tf.train.AdamOptimizer)

        hparams = {
            "type": "MomentumOptimizer",
            "kwargs": {
                "learning_rate": 0.001,
                "momentum": 0.9,
                "use_nesterov": True
            }
        }
        momentum_optimizer_fn, _ = opt.get_optimizer_fn(hparams)
        momentum_optimizer = momentum_optimizer_fn()
        self.assertIsInstance(momentum_optimizer, tf.train.MomentumOptimizer)

        hparams = {
            "type": tf.train.MomentumOptimizer,
            "kwargs": {
                "momentum": 0.9,
                "use_nesterov": True
            }
        }
        momentum_optimizer_fn, _ = opt.get_optimizer_fn(hparams)
        momentum_optimizer = momentum_optimizer_fn(0.001)
        self.assertIsInstance(momentum_optimizer, tf.train.MomentumOptimizer)

        hparams = {
            "type": tf.train.MomentumOptimizer(0.001, 0.9)
        }
        momentum_optimizer, _ = opt.get_optimizer_fn(hparams)
        self.assertIsInstance(momentum_optimizer, tf.train.MomentumOptimizer)


    def test_get_learning_rate_decay_fn(self): # pylint: disable=too-many-locals
        """Tests get_learning_rate_decay_fn.
        """
        default_lr_decay_fn = opt.get_learning_rate_decay_fn(
            opt.default_optimization_hparams()["learning_rate_decay"])
        self.assertIsNone(default_lr_decay_fn)

        boundaries = [2, 4]
        values = [0.1, 0.01, 0.001]
        hparams = {
            "type": "piecewise_constant",
            "kwargs": {
                "boundaries": boundaries,
                "values": values
            },
            "min_learning_rate": 0.05,
            "start_decay_step": 1,
            "end_decay_step": utils.MAX_SEQ_LENGTH,
        }
        pc_lr_decay_fn = opt.get_learning_rate_decay_fn(hparams)

        global_step = 1
        pc_lr = pc_lr_decay_fn(learning_rate=1., global_step=global_step)
        pc_lr_true = tf.train.piecewise_constant(
            global_step-hparams["start_decay_step"], boundaries, values)

        hparams["type"] = "natural_exp_decay"
        hparams["kwargs"] = {
            "decay_steps": 1,
            "decay_rate": 0.5
        }
        ned_lr_decay_fn = opt.get_learning_rate_decay_fn(hparams)
        ned_lr = ned_lr_decay_fn(learning_rate=1., global_step=global_step)
        ned_lr_true = tf.train.natural_exp_decay(
            1., global_step-hparams["start_decay_step"],
            hparams["kwargs"]["decay_steps"], hparams["kwargs"]["decay_rate"])

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            pc_lr_, pc_lr_true_, ned_lr_, ned_lr_true_ = sess.run(
                [pc_lr, pc_lr_true, ned_lr, ned_lr_true])
            self.assertEqual(pc_lr_, pc_lr_true_)
            self.assertEqual(ned_lr_, ned_lr_true_)

    def test_get_gradient_clip_fn(self):    # pylint: disable=too-many-locals
        """Tests get_gradient_clip_fn.
        """
        default_grad_clip_fn = opt.get_gradient_clip_fn(
            opt.default_optimization_hparams()["gradient_clip"])
        self.assertIsNone(default_grad_clip_fn)

        grads = [tf.random_uniform([10, 10], -1., 1.) for _ in range(5)]
        grads_and_vars = list(zip(grads, range(5)))

        hparams = {
            "type": "clip_by_global_norm",
            "kwargs": {
                "clip_norm": 0.1
            }
        }
        gn_grad_clip_fn = opt.get_gradient_clip_fn(hparams)
        gn_grads_and_vars = gn_grad_clip_fn(grads_and_vars)
        gn_grads, _ = zip(*gn_grads_and_vars)
        gn_grads_true, _ = tf.clip_by_global_norm(
            grads, hparams["kwargs"]["clip_norm"])

        hparams = {
            "type": "clip_by_value",
            "kwargs": {
                "clip_value_min": -0.01,
                "clip_value_max": 0.01
            }
        }
        v_grad_clip_fn = opt.get_gradient_clip_fn(hparams)
        v_grads_and_vars = v_grad_clip_fn(grads_and_vars)
        v_grads, _ = zip(*v_grads_and_vars)
        v_grads_true = tf.clip_by_value(grads,
                                        hparams["kwargs"]["clip_value_min"],
                                        hparams["kwargs"]["clip_value_max"])

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            gn_grads_, gn_grads_true_, v_grads_, v_grads_true_ = sess.run(
                [gn_grads, gn_grads_true, v_grads, v_grads_true])
            np.testing.assert_array_equal(gn_grads_, gn_grads_true_)
            np.testing.assert_array_equal(v_grads_, v_grads_true_)

    def test_get_train_op(self):
        """Tests get_train_op.
        """
        var = tf.Variable(0.)
        loss = tf.nn.l2_loss(var)
        train_op = opt.get_train_op(loss)
        self.assertTrue(tf.contrib.framework.is_tensor(train_op))

if __name__ == "__main__":
    tf.test.main()
