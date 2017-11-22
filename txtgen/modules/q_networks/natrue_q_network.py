from txtgen.modules.q_networks import QNetworksBase
# from txtgen.losses.dqn_losses import l2_loss
# from txtgen.core import optimization as opt
from txtgen.modules.q_networks.basic_components import *

import numpy as np


class NatrueQNetwork(QNetworksBase):
    def __init__(self, input_dimension, output_dimension, hparams=None):
        QNetworksBase.__init__(self, hparams=hparams)
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        if hparams is None:
            hparams = NatrueQNetwork.default_hparams()
        hparams['loss_fn'] = l2_loss
        self.qnet = MLPNetwork(input_dimension=input_dimension, output_dimension=output_dimension,
                               hidden_list=hparams['hidden_list'], loss_fn=hparams['loss_fn'],
                               train_hparams=hparams['train_hparams'])
        self.target = MLPNetwork(input_dimension=input_dimension, output_dimension=output_dimension,
                                 hidden_list=hparams['hidden_list'], loss_fn=hparams['loss_fn'],
                                 train_hparams=hparams['train_hparams'])
        self.update_target()
        self.time_step = 0
        self.update_period = hparams['update_period']
        self.gamma = hparams['gamma']

    def _build(self, *args, **kwargs):
        return

    @staticmethod
    def default_hparams():
        return {
            'name': 'nature_q_network',
            'network_type': 'mlp_network',  # 'dueling_network'
            'hidden_list': [128, 128],
            # 'loss_fn': l2_loss,
            'train_hparams': opt.default_optimization_hparams(),
            'update_period': 100,
            'gamma': 0.99
        }

    def update_target(self):
        self.target.set_params(self.qnet.get_params())

    def train(self, mini_batch=None):
        state_batch = np.array([u['state'] for u in mini_batch])
        action_batch = np.array([u['action'] for u in mini_batch])
        reward_batch = np.array([u['reward'] for u in mini_batch])
        next_state_batch = np.array([u['next_state'] for u in mini_batch])
        is_terminal_batch = np.array([u['is_terminal'] for u in mini_batch])

        qvalue = self.target.get_qvalue(state_batch=next_state_batch)
        y_batch = reward_batch
        for i in range(len(mini_batch)):
            if not is_terminal_batch[i]:
                y_batch[i] += self.gamma * np.max(qvalue[i])

        self.qnet.train(feed_dict={
            'state_input:0': state_batch,
            'y_input:0': y_batch,
            'action_input:0': action_batch
        })
        self.time_step += 1
        if self.time_step % self.update_period == 0:
            self.update_target()

    def get_qvalue(self, state_batch):
        return self.qnet.get_qvalue(state_batch=state_batch)
