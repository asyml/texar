# Copyright 2018 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Unit tests for sequence prediction policy gradient agents.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf

from texar.modules.decoders.rnn_decoders import BasicRNNDecoder
from texar.agents import SeqPGAgent
from texar import context

class SeqPGAgentTest(tf.test.TestCase):
    """Tests :class:`texar.agents.SeqPGAgent`
    """

    def setUp(self):
        tf.test.TestCase.setUp(self)
        self._vocab_size = 4
        self._max_time = 8
        self._batch_size = 16
        self._emb_dim = 20
        self._inputs = tf.random_uniform(
            [self._batch_size, self._max_time, self._emb_dim],
            maxval=1., dtype=tf.float32)
        self._embedding = tf.random_uniform(
            [self._vocab_size, self._emb_dim], maxval=1., dtype=tf.float32)

    def test_seq_pg_agent(self):
        """Tests logits.
        """
        decoder = BasicRNNDecoder(vocab_size=self._vocab_size)
        outputs, _, sequence_length = decoder(
            decoding_strategy="infer_greedy",
            max_decoding_length=10,
            embedding=self._embedding,
            start_tokens=[1]*self._batch_size,
            end_token=2)

        agent = SeqPGAgent(
            outputs.sample_id, outputs.logits, sequence_length,
            decoder.trainable_variables)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            agent.sess = sess

            feed_dict = {context.global_mode(): tf.estimator.ModeKeys.TRAIN}
            for _ in range(2):
                vals = agent.get_samples(feed_dict=feed_dict)
                self.assertEqual(vals['samples'].shape[0], self._batch_size)

                loss_1 = agent.observe([1.]*self._batch_size)
                loss_2 = agent.observe(
                    [1.]*self._batch_size, train_policy=False)
                self.assertEqual(loss_1.shape, ())
                self.assertEqual(loss_2.shape, ())

if __name__ == "__main__":
    tf.test.main()
