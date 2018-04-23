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

        inputs = tf.random_uniform(shape=[64, 4])
        outputs = policy(inputs=inputs)
        self.assertEqual(outputs['action'].shape, outputs['log_prob'].shape)
        self.assertIsInstance(
            outputs['distribution'], tf.distributions.Categorical)

if __name__ == "__main__":
    tf.test.main()
