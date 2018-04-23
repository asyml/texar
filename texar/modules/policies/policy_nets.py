"""Policy models based on feed forward networks.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf

from texar.module_base import ModuleBase
from texar.modules.networks.network_base import FeedForwardNetworkBase
from texar.agents.agent_utils import Space
from texar.utils import utils

# pylint: disable=no-member

__all__ = [
    'PolicyNetBase',
    'CategoricalPolicyNet'
]

def _build_network(network, kwargs, network_type, network_hparams):
    if network is not None:
        network = network
    else:
        kwargs = {'hparams': network_hparams}
        kwargs.update(kwargs or {})
        network = utils.check_or_get_instance(
            network_type,
            kwargs,
            module_paths=['texar.modules', 'texar.custom'],
            classtype=FeedForwardNetworkBase)
    return network

class PolicyNetBase(ModuleBase):
    """Policy model based on feed forward network.
    """
    def __init__(self,
                 network=None,
                 network_kwargs=None,
                 hparams=None):
        ModuleBase.__init__(self, hparams=hparams)

        with tf.variable_scope(self.variable_scope):
            self._network = _build_network(
                network, network_kwargs, self._hparams.network_type,
                self._hparams.network_hparams.todict())

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        TODO
        """
        return {
            'network_type': 'FeedForwardNetwork',
            'network_hparams': {
                'layers': [
                    {'type': 'Dense',
                     'kwargs': {'units': 256, 'activation': 'relu'}},
                    {'type': 'Dense',
                     'kwargs': {'units': 256, 'activation': 'relu'}},
                ]
            },
            'distribution_kwargs': None,
            'name': 'policy_net',
            '@no_typecheck': ['network_type', 'network_hparams']
        }

    def _build(self, inputs, mode=None): # pylint: disable=arguments-differ
        raise NotImplementedError
        #output = self.network(inputs)
        #if not self._built:
        #    self._add_internal_trainable_variables()
        #    self._add_trainable_variable(self._network.trainable_variables)
        #    self._built = True

        #return output

    @property
    def network(self):
        """The network.
        """
        return self._network


#TODO(zhiting): Allow structured discrete actions.
class CategoricalPolicyNet(PolicyNetBase):
    """Policy net with Categorical distribution for discrete actions.
    """

    def __init__(self,
                 action_space=None,
                 network=None,
                 network_kwargs=None,
                 hparams=None):
        PolicyNetBase.__init__(self, hparams=hparams)

        with tf.variable_scope(self.variable_scope):
            self._append_output_layer(action_space)

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        TODO
        """
        hparams = PolicyNetBase.default_hparams()
        hparams.update({
            'distribution_kwargs': {
                'dtype': 'int32',
                'validate_args': False,
                'allow_nan_stats': True
            },
            'action_space': 2,
            'make_output_layer': True
        })
        return hparams

    def _append_output_layer(self, action_space):
        if not self._hparams.make_output_layer:
            return

        if action_space is None:
            action_space = Space(
                low=0, high=self._hparams.action_space, dtype=np.int32)
        if action_space.shape != ():
            raise ValueError('Only scalar discrete action is supported.')
        else:
            output_size = action_space.high - action_space.low

        layer_hparams = {
            'type': 'Dense',
            'kwargs': {'units': output_size}
        }
        self._network.append_layer(layer_hparams)

    def _build(self, inputs, mode=None):
        logits = self._network(inputs, mode=mode)

        dkwargs = self._hparams.distribution_kwargs.todict()
        dkwargs['dtype'] = utils.get_tf_dtype(dkwargs['dtype'])
        dstr = tf.distributions.Categorical(logits=logits, **dkwargs)

        action = dstr.sample()
        log_prob = dstr.log_prob(action)
        outputs = dict(
            action=action,
            log_prob=log_prob,
            distribution=dstr
        )

        if not self._built:
            self._add_internal_trainable_variables()
            self._add_trainable_variable(self._network.trainable_variables)
            self._built = True

        return outputs
