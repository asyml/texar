"""
Deep Q Network/Policy Gradient/Actor-Critic for the CartPole game in OpenAI gym.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name
# pylint: disable=wildcard-import

#import numpy as np

import gym
#import tensorflow as tf

from texar.agents.pg_agent import PGAgent
#from texar.agents.ac_agent import *

env = gym.make('CartPole-v0')
env = env.unwrapped

if __name__ == '__main__':
    # hparams = ACAgent.default_hparams()
    # hparams['actor_network'] = {
    #     'hparams': {
    #         'network_hparams': {
    #             'layers': [
    #                 {
    #                     'type': 'Dense',
    #                     'kwargs': {
    #                         'units': 50,
    #                         'activation': 'relu'
    #                     }
    #                 }, {
    #                     'type': 'Dense',
    #                     'kwargs': {
    #                         'units': 2
    #                     }
    #                 }
    #             ]
    #         }
    #     }
    # }
    # hparams['critic_network'] = {
    #     'hparams': {
    #         'network_hparams': {
    #             'layers': [
    #                 {
    #                     'type': 'Dense',
    #                     'kwargs': {
    #                         'units': 20,
    #                         'activation': 'relu'
    #                     }
    #                 }, {
    #                     'type': 'Dense',
    #                     'kwargs': {
    #                         'units': 1
    #                     }
    #                 }
    #             ]
    #         }
    #     }
    # }
    hparams = PGAgent.default_hparams()
    hparams['network'] = {
        'hparams': {
            'network_hparams': {
                'layers': [
                    {
                        'type': 'Dense',
                        'kwargs': {
                            'units': 256,
                            'activation': 'relu'
                        }
                    }, {
                        'type': 'Dense',
                        'kwargs': {
                            'units': 256,
                            'activation': 'relu'
                        }
                    }, {
                        'type': 'Dense',
                        'kwargs': {
                            'units': 2
                        }
                    }
                ]
            }
        }
    }
    agent = PGAgent(actions=2, state_shape=(4, ), hparams=hparams)

    for i in range(5000):
        reward_sum = 0.0
        observation = env.reset()
        agent.set_initial_state(observation=observation)
        while True:
            action_id = agent.get_action()

            next_observation, reward, is_terminal, info = \
                env.step(action=action_id)
            if is_terminal:
                reward = -20
            agent.perceive(next_observation=next_observation,
                           action_id=action_id,
                           reward=reward, is_terminal=is_terminal)

            reward_sum += reward
            if is_terminal:
                break
        print('episode {round_id}: {reward}'.format(round_id=i,
                                                    reward=reward_sum))
