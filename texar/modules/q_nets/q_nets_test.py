"""
Unit tests for RNN decoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import tensorflow as tf

from texar.modules import NatureQNet

class NatureQNetTest(tf.test.TestCase):
    """Tests :class:`texar.modules.NatureQNet`
    """

    def test_nature_q_net(self):
        """Tests logics.
        """
        hparams = NatureQNet.default_hparams()
        hparams['network_hparams']['layers'] = [
            {
                'type': 'Dense',
                'kwargs': {
                    'units': 128,
                    'activation': 'relu'
                }
            },
            {
                'type': 'Dense',
                'kwargs': {
                    'units': 2
                }
            }
        ]
        network = NatureQNet(hparams=hparams)

        inputs = tf.zeros(shape=[64, 10])
        outputs = network(inputs)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            shapes = sess.run([tf.shape(output) for output in outputs])
            np.testing.assert_array_equal(shapes[0], shapes[1])


if __name__ == "__main__":
    tf.test.main()
