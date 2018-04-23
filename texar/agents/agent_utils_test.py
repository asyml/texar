#
"""
Unit tests for agent utilities.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# pylint: disable=no-member, invalid-name, too-many-arguments

import numpy as np

import tensorflow as tf

from texar.agents.agent_utils import Space

class SpaceTest(tf.test.TestCase):
    """Tests the Space class.
    """

    def _test_space(self, s, shape, low, high, dtype):
        self.assertEqual(s.shape, shape)
        self.assertEqual(s.low, low)
        self.assertEqual(s.high, high)
        self.assertEqual(s.dtype, dtype)

    def test_space(self):
        """Tests descrete space.
        """
        s = Space(shape=(), low=0, high=10, dtype=np.int32)
        self._test_space(s, (), 0, 10, np.dtype(np.int32))
        self.assertTrue(s.contains(5))
        self.assertFalse(s.contains(5.))
        self.assertFalse(s.contains(15))

        s = Space(low=0, high=10, dtype=np.int32)
        self._test_space(s, (), 0, 10, np.dtype(np.int32))


if __name__ == "__main__":
    tf.test.main()
