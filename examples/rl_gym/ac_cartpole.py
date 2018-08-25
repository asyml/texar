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
Policy gradient for the CartPole game in OpenAI gym.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name

import importlib
import gym
import tensorflow as tf
import texar as tx

flags = tf.flags

flags.DEFINE_string("config", "config", "The config to use.")

FLAGS = flags.FLAGS

config = importlib.import_module(FLAGS.config)

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env = env.unwrapped

    env_config = tx.agents.get_gym_env_config(env)

    agent = tx.agents.ActorCriticAgent(env_config=env_config)
    with tf.Session() as sess:
        agent.sess = sess

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        feed_dict = {tx.global_mode(): tf.estimator.ModeKeys.TRAIN}

        for e in range(5000):
            reward_sum = 0.
            observ = env.reset()
            agent.reset()
            while True:
                action = agent.get_action(observ, feed_dict=feed_dict)

                next_observ, reward, terminal, _ = env.step(action=action)
                agent.observe(reward, terminal, feed_dict=feed_dict)
                observ = next_observ

                reward_sum += reward
                if terminal:
                    break

            if (e + 1) % 10 == 0:
                print('episode {}: {}'.format(e + 1, reward_sum))
