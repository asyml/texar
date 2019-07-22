"""
Unit tests for xlnet model utils.
"""
import tensorflow as tf

from texar.modules.pretrained.xlnet_model_utils import \
    PositionWiseFF, RelativePositionalEncoding, RelativeMutiheadAttention


class XLNetModelUtilsTest(tf.test.TestCase):
    r"""Tests xlnet model utils.
    """

    def test_PositionWiseFF(self):

        # Case 1
        model = PositionWiseFF()
        inputs = tf.random_uniform(shape=(32, model.hparams.hidden_dim))
        outputs = model(inputs)
        self.assertEqual(outputs.shape, [32, model._hparams.hidden_dim])

        # Case 2
        hparams = {
            "hidden_dim": 16,
            "ffn_inner_dim": 32,
            "dropout": 0.1,
            "activation": 'relu',
        }
        model = PositionWiseFF(hparams=hparams)
        inputs = tf.random_uniform(shape=(32, 16))
        outputs = model(inputs)
        self.assertEqual(outputs.shape, [32, 16])

        # Case 3
        hparams = {
            "hidden_dim": 16,
            "ffn_inner_dim": 32,
            "dropout": 0.1,
            "activation": 'gelu',
        }
        model = PositionWiseFF(hparams=hparams)
        inputs = tf.random_uniform(shape=(32, 16))
        outputs = model(inputs)
        self.assertEqual(outputs.shape, [32, 16])

    def test_RelativeMultiheadAttention(self):

        model = RelativeMutiheadAttention()

        states_h = tf.random_uniform(shape=(16, 32, model._hparams.hidden_dim))
        pos_embed = tf.random_uniform(shape=(24, 32, model._hparams.hidden_dim))

        output_h, output_g = model(states_h=states_h, pos_embed=pos_embed)

        self.assertEqual(output_h.shape,
                         [16, 32, model._hparams.hidden_dim])
        self.assertEqual(output_g, None)

    def test_RelativePositionalEncoding(self):

        batch_size = 16
        seq_len = 8
        total_len = 32

        # Case 1
        model = RelativePositionalEncoding()
        pos_embed = model(batch_size=batch_size,
                          seq_len=seq_len,
                          total_len=total_len)
        self.assertEqual(pos_embed.shape,
                         [40, 16, model._hparams.dim])

        # Case 2
        model = RelativePositionalEncoding()
        pos_embed = model(batch_size=batch_size,
                          seq_len=seq_len,
                          total_len=total_len,
                          attn_type='uni')
        self.assertEqual(pos_embed.shape,
                         [33, 16, model._hparams.dim])


if __name__ == "__main__":
    tf.test.main()
