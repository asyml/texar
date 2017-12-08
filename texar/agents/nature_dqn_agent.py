from agent_base import AgentBase
from texar.core import optimization as opt
from texar.core import get_class
from texar.losses.dqn_losses import l2_loss

import tensorflow as tf
import numpy as np
import random


class NatureDQNAgent(AgentBase):
    def __init__(self, actions, state_shape, hparams=None):
        AgentBase.__init__(self, hparams=hparams)
        self.actions = actions
        self.state_shape = state_shape

        self.batch_size = self._hparams.batch_size
        self.discount_factor = self._hparams.discount_factor
        self.observation_steps = self._hparams.observation_steps
        self.update_period = self._hparams.update_period

        # network
        network_type = get_class(class_name=self._hparams.qnetwork.type,
                                 module_paths=['texar.modules', 'texar.custom'])
        self.network = network_type(self._hparams.qnetwork.hparams)

        # replay_memory
        replay_memory_type = get_class(class_name=self._hparams.replay_memory.type,
                                       module_paths=['texar.core', 'texar.custom'])
        self.replay_memory = replay_memory_type(self._hparams.replay_memory.hparams)

        # loss && trainer
        with tf.variable_scope(self.network.variable_scope):
            self.state_input = tf.placeholder(dtype=tf.float64, shape=[None, ] + list(state_shape))
            self.y_input = tf.placeholder(dtype=tf.float64, shape=(None, ))
            self.action_input = tf.placeholder(dtype=tf.float64, shape=(None, self.actions))

            self.qnet_qvalue, self.target_qvalue = self.network(self.state_input)
            self.loss = self._hparams.trainer.loss_fn(
                qvalue=self.qnet_qvalue, action_input=self.action_input, y_input=self.y_input)
            self.trainer = opt.get_train_op(loss=self.loss, variables=None,
                                            hparams=self._hparams.trainer.optimization_hparams)

        # exploration
        exploration_type = get_class(class_name=self._hparams.exploration.type,
                                     module_paths=['texar.core', 'texar.custom'])
        self.exploration = exploration_type(self._hparams.exploration.hparams)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.update_target()

    @staticmethod
    def default_hparams():
        return {
            'name': 'nature_dqn_agent',
            'batch_size': 32,
            'discount_factor': 0.99,
            'observation_steps': 100,
            'update_period': 100,
            'qnetwork': {
                'type': 'NatureQNetwork',
                'hparams': None
            },
            'replay_memory': {
                'type': 'DequeReplayMemory',
                'hparams': None
            },
            'trainer': {
                'loss_fn': l2_loss,
                'optimization_hparams': opt.default_optimization_hparams(),
            },
            'exploration': {
                'type': 'EpsilonDecayExploration',
                'hparams': None
            }
        }

    def set_initial_state(self, observation):
        self.current_state = np.array(observation)

    def perceive(self, action, reward, is_terminal, next_observation):
        self.replay_memory.push({
            'state': self.current_state,
            'action': action,
            'reward': reward,
            'is_terminal': is_terminal,
            'next_state': next_observation
        })
        self.timestep += 1
        self.current_state = np.array(next_observation)

        if self.timestep > self.observation_steps:
            self.train_qnet()

    def train_qnet(self):
        minibatch = self.replay_memory.sample(self.batch_size)
        state_batch = np.array([data['state'] for data in minibatch])
        action_batch = np.array([data['action'] for data in minibatch])
        reward_batch = np.array([data['reward'] for data in minibatch])
        is_terminal_batch = np.array([data['is_terminal'] for data in minibatch])
        next_state_batch = np.array([data['next_state'] for data in minibatch])

        qvalue = self.sess.run(self.target_qvalue, feed_dict={self.state_input: next_state_batch})
        y_batch = reward_batch
        for i in range(self.batch_size):
            if not is_terminal_batch[i]:
                y_batch[i] += self.discount_factor * np.max(qvalue[i])

        self.sess.run(self.trainer, feed_dict={
            self.state_input: state_batch,
            self.y_input: y_batch,
            self.action_input: action_batch
        })

        if self.timestep % self.update_period == 0:
            self.update_target()

    def update_target(self):
        variable_dict = {}
        for key in self.network.qnet.trainable_variables:
            variable_dict['/'.join(key.name.split('/')[2:])] = self.sess.run(key)
        for key in self.network.target.trainable_variables:
            if '/'.join(key.name.split('/')[2:]) in variable_dict:
                self.sess.run(tf.assign(key, variable_dict['/'.join(key.name.split('/')[2:])]))
            else:
                raise ValueError

    def get_action(self, state=None, action_mask=None):
        if state is None:
            state = self.current_state
        if action_mask is None:
            action_mask = [True for _ in range(self.actions)]

        qvalue = self.sess.run(self.qnet_qvalue, feed_dict={self.state_input: np.array([state])})
        action = np.zeros(shape=(self.actions,))
        if random.random() < self.exploration.epsilon():
            while True:
                action_id = random.randrange(self.actions)
                if action_mask[action_id]:
                    break
        else:
            for i in range(self.actions):
                if not action_mask[i]:
                    qvalue[i] -= 10.0 ** 9
            action_id = np.argmax(qvalue)
        action[action_id] = 1.0

        self.exploration.add_timestep()
        return action
