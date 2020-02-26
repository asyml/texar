"""
Unit tests for GPT2 decoder.
"""

import tensorflow as tf

from texar.tf.modules.decoders.gpt2_decoder import GPT2Decoder
from texar.tf.modules.decoders.transformer_decoders import \
    TransformerDecoderOutput
from texar.tf.utils.test import pretrained_test


class GPT2DecoderTest(tf.test.TestCase):
    r"""Tests :class:`~texar.torch.modules.GPT2Decoder`
    """

    @pretrained_test
    def test_hparams(self):
        r"""Tests the priority of the decoder arch parameters.
        """

        inputs = tf.placeholder(dtype=tf.int32, shape=[2, 3])

        # case 1: set "pretrained_mode_name" by constructor argument
        hparams = {
            "pretrained_model_name": "gpt2-medium",
        }
        decoder = GPT2Decoder(pretrained_model_name="gpt2-small",
                              hparams=hparams)
        _ = decoder(inputs=inputs)
        self.assertEqual(decoder.hparams.decoder.num_blocks, 12)

        # case 2: set "pretrained_mode_name" by hparams
        hparams = {
            "pretrained_model_name": "gpt2-small",
            "decoder": {
                "num_blocks": 6,
            }
        }
        decoder = GPT2Decoder(hparams=hparams)
        _ = decoder(inputs=inputs)
        self.assertEqual(decoder.hparams.decoder.num_blocks, 12)

        # case 3: set to None in both hparams and constructor argument
        hparams = {
            "pretrained_model_name": None,
            "decoder": {
                "num_blocks": 6,
            }
        }
        decoder = GPT2Decoder(hparams=hparams)
        _ = decoder(inputs=inputs)
        self.assertEqual(decoder.hparams.decoder.num_blocks, 6)

        # case 4: using default hparams
        decoder = GPT2Decoder()
        _ = decoder(inputs=inputs)
        self.assertEqual(decoder.hparams.decoder.num_blocks, 12)

    @pretrained_test
    def test_trainable_variables(self):
        r"""Tests the functionality of automatically collecting trainable
        variables.
        """

        inputs = tf.placeholder(dtype=tf.int32, shape=[2, 3])

        def get_variable_num(n_layers: int) -> int:
            return 1 + 1 + n_layers * 16 + 2

        # case 1: GPT2 small
        decoder = GPT2Decoder()
        _ = decoder(inputs=inputs)
        self.assertEqual(len(decoder.trainable_variables), get_variable_num(12))

        # case 2: GPT2 medium
        hparams = {
            "pretrained_model_name": "gpt2-medium",
        }
        decoder = GPT2Decoder(hparams=hparams)
        _ = decoder(inputs=inputs)
        self.assertEqual(len(decoder.trainable_variables), get_variable_num(24))

        # case 2: GPT2 large
        hparams = {
            "pretrained_model_name": "gpt2-large",
        }
        decoder = GPT2Decoder(hparams=hparams)
        _ = decoder(inputs=inputs)
        self.assertEqual(len(decoder.trainable_variables), get_variable_num(36))

        # case 3: self-designed GPT2
        hparams = {
            "pretrained_model_name": None,
            "decoder": {
                "num_blocks": 6,
            }
        }
        decoder = GPT2Decoder(hparams=hparams)
        _ = decoder(inputs=inputs)
        self.assertEqual(len(decoder.trainable_variables), get_variable_num(6))

    def test_decode_train(self):
        r"""Tests train_greedy.
        """
        hparams = {
            "pretrained_model_name": None
        }
        decoder = GPT2Decoder(hparams=hparams)

        max_time = 8
        batch_size = 16
        inputs = tf.random_uniform([batch_size, max_time],
                                   maxval=50257, dtype=tf.int32)
        outputs = decoder(inputs=inputs)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs_ = sess.run(outputs)
            self.assertEqual(outputs_.logits.shape, (batch_size,
                                                     max_time, 50257))
            self.assertEqual(outputs_.sample_id.shape, (batch_size, max_time))

    def test_decode_infer_greedy(self):
        r"""Tests infer_greedy
        """
        hparams = {
            "pretrained_model_name": None
        }
        decoder = GPT2Decoder(hparams=hparams)

        start_tokens = tf.fill([16], 1)
        end_token = 2
        outputs, length = decoder(max_decoding_length=4,
                                  start_tokens=start_tokens,
                                  end_token=end_token,
                                  decoding_strategy="infer_greedy")

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs_ = sess.run(outputs)
            self.assertIsInstance(outputs_, TransformerDecoderOutput)


if __name__ == "__main__":
    tf.test.main()
