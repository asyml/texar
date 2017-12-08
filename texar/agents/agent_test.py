from nature_dqn_agent import NatureDQNAgent


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
agent1 = NatureDQNAgent(actions=2, state_shape=(4, ), hparams=hparams)
agent2 = NatureDQNAgent(actions=2, state_shape=(4, ), hparams=hparams)
