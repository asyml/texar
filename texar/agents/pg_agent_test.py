"""
Unit tests for RNN decoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import tensorflow as tf

from texar.modules import PGNet

class PGAgentTest(tf.test.TestCase):
    """Tests :class:`texar.modules.PGNet`
    """

    def test_pg_agent(self):
        """Tests logics.
        """
        hparams = PGNet.default_hparams()
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
                    'units': 10,
                    'activation': 'softmax'
                }
            }
        ]
        network = PGNet(hparams=hparams)

        inputs = tf.zeros(shape=[64, 10])
        outputs = network(inputs)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            shapes = sess.run(outputs)
            print(shapes)

if __name__ == "__main__":
    tf.test.main()
