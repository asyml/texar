"""
Unit tests for average recoder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from texar.utils.average_recorder import _SingleAverageRecorder, AverageRecorder


class AverageRecorderTest(tf.test.TestCase):
    """Tests average recoder.
    """

    def test_single_average_recoder(self):
        """Tests :class:`~texar.utils._SingleAverageRecorder`
        """
        recoder = _SingleAverageRecorder(5)
        for i in range(100):
            self.assertEqual(recoder.add(1), 1.)
            self.assertEqual(recoder.avg(), 1.)

        recoder = _SingleAverageRecorder()
        for i in range(100):
            self.assertEqual(recoder.add(1), 1.)
            self.assertEqual(recoder.avg(), 1.)

    def test_average_recorder(self):
        """Tests :class:`~texar.utils.AverageRecorder`
        """
        recorder = AverageRecorder(5)
        for i in range(100):
            self.assertEqual(recorder.add([1., 2.]), [1., 2.])
            self.assertEqual(recorder.add([1.]), [1., 2.])
            self.assertEqual(recorder.avg(), [1., 2.])
            self.assertEqual(recorder.avg(0), [1.])
            self.assertEqual(recorder.avg(1), [2.])
            self.assertEqual(recorder.avg([0, 1]), [1., 2.])

        recorder = AverageRecorder()
        for i in range(100):
            self.assertEqual(recorder.add({'1': 1, '2': 2}), {'1': 1., '2': 2.})
            self.assertEqual(recorder.add({'1': 1}), {'1': 1., '2': 2.})
            self.assertEqual(recorder.avg(), {'1': 1., '2': 2.})
            self.assertEqual(recorder.avg('1'), {'1': 1.})
            self.assertEqual(recorder.avg('2'), {'2': 2.})
            self.assertEqual(recorder.avg(['1', '2']), {'1': 1., '2': 2.})

if __name__ == "__main__":
    tf.test.main()

