"""
Test for Agents
"""

from texar.agents.nature_dqn_agent import NatureDQNAgent


hparams = NatureDQNAgent.default_hparams()
hparams['qnetwork'] = {
    'hparams': {
        'network_hparams': {
            'layers': [
                {
                    'type': 'Dense',
                    'kwargs': {
                        'units': 128,
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
agent_1 = NatureDQNAgent(actions=2, state_shape=(4, ), hparams=hparams)
agent_2 = NatureDQNAgent(actions=2, state_shape=(4, ), hparams=hparams)
