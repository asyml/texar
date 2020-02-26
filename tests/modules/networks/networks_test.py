"""
Unit tests for feed forward neural networks.
"""

import tensorflow as tf

from texar.tf.modules.networks.networks import FeedForwardNetwork

# pylint: disable=no-member, invalid-name


class FeedForwardNetworkTest(tf.test.TestCase):
    """Tests the class
    :class:`~texar.tf.modules.networks.networks.FeedForwardNetwork`.
    """

    def test_feedforward(self):
        """Tests feed-forward.
        """
        hparams = {
            "layers": [
                {
                    "type": "Dense",
                },
                {
                    "type": "Dense",
                }
            ]
        }

        nn = FeedForwardNetwork(hparams=hparams)

        self.assertEqual(len(nn.layers), len(hparams["layers"]))
        _ = nn(tf.ones([64, 16, 16]))
        self.assertEqual(len(nn.trainable_variables),
                         len(hparams["layers"]) * 2)
        self.assertEqual(len(nn.layer_outputs), len(hparams["layers"]))


if __name__ == "__main__":
    tf.test.main()
