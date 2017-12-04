from q_networks import NatureQNetwork

import tensorflow as tf

hparams = NatureQNetwork.default_hparams()
hparams['network_hparams']['layers'] = [
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
network = NatureQNetwork(hparams=hparams)

print network(tf.placeholder(dtype=tf.float64, shape=[None, 10]))


