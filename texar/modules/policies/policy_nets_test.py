#
"""
Tests policy nets.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf

from texar.modules.policies.policy_nets import CategoricalPolicyNet

class CategoricalPolicyNetTest(tf.test.TestCase):
    """Tests :class:`texar.modules.CategoricalPolicyNet`.
    """

    def test_categorical_policy(self):
        """Tests logics.
        """
        policy = CategoricalPolicyNet()

        inputs = tf.random_uniform(shape=[1, 4])
        outputs = policy(inputs=inputs)
        self.assertEqual(list(outputs['action'].shape[1:]),
                         list(policy.action_space.shape))
        self.assertIsInstance(outputs['dist'],
                              tf.distributions.Categorical)


        inputs = tf.random_uniform(shape=[64, 4])
        outputs = policy(inputs=inputs)
        self.assertEqual(list(outputs['action'].shape[1:]),
                         list(policy.action_space.shape))
        self.assertEqual(int(outputs['action'].shape[0]), 64)

if __name__ == "__main__":
    tf.test.main()
