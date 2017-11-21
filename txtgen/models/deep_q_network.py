import txtgen
import numpy as np
import random


class DeepQNetwork:
    def __init__(self, actions, state_dimension, hparams=None):
        if hparams is None:
            hparams = self.default_hparams()
        self.actions = actions
        self.state_dimension = state_dimension
        self.observe_steps = hparams['observe_steps']
        self.explore_steps = hparams['explore_steps']
        self.initial_epsilon = hparams['initial_epsilon']
        self.final_epsilon = hparams['final_epsilon']
        self.batch_size = hparams['batch_size']
        self.network = hparams['network_type'](input_dimension=state_dimension, output_dimension=actions,
                                               hparams=hparams['network_hparams'])
        self.replay_memory = hparams['replay_memory_type'](hparams=hparams['replay_memory_hparams'])

        self.current_state = None
        self.cnt_steps = 0
        self.epsilon = self.initial_epsilon

    @staticmethod
    def default_hparams():
        return {
            'name': 'deep_q_network',
            'observe_steps': 100,
            'explore_steps': 20000,
            'initial_epsilon': 0.1,
            'final_epsilon': 0.0,
            'batch_size': 32,
            'network_type': txtgen.modules.NatrueQNetwork,
            'network_hparams': txtgen.modules.NatrueQNetwork.default_hparams(),
            'replay_memory_type': txtgen.modules.DequeReplayMemory,
            'replay_memory_hparams': txtgen.modules.DequeReplayMemory.default_hparams()
        }

    def set_initial_state(self, observation):
        self.current_state = np.array(observation)

    def perceive(self, next_observation, action, reward, is_terminal):
        new_state = np.array(next_observation)
        self.replay_memory.push({
            'state': self.current_state,
            'action': action,
            'reward': reward,
            'next_state': new_state,
            'is_terminal': is_terminal
        })
        if self.cnt_steps > self.observe_steps:
            self.train_network()
        self.cnt_steps += 1
        self.current_state = new_state

    def train_network(self):
        mini_batch = self.replay_memory.sample(size=self.batch_size)
        self.network.train(mini_batch=mini_batch)

    def get_action(self, action_mask=None):
        if action_mask is None:
            action_mask = [True for i in range(self.actions)]

        qvalue = self.network.get_qvalue(state_batch=np.array([self.current_state]))
        action = np.zeros(shape=(self.actions,))
        action_id = 0
        if random.random() < self.epsilon:
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

        if self.epsilon > self.final_epsilon and self.cnt_steps > self.observe_steps:
            self.epsilon -= (self.initial_epsilon - self.final_epsilon) / self.explore_steps

        return action
