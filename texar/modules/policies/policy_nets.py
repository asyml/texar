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
"""Policy models based on feed forward networks.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf

from texar.module_base import ModuleBase
from texar.agents.agent_utils import Space
from texar.utils import utils
from texar.utils.dtypes import get_tf_dtype

# pylint: disable=no-member

__all__ = [
    'PolicyNetBase',
    'CategoricalPolicyNet'
]

class PolicyNetBase(ModuleBase):
    """Policy net that takes in states and outputs actions.

    Args:
        network (optional): A network that takes in state and returns
            outputs for generating actions. For example, an instance of subclass
            of :class:`~texar.modules.FeedForwardNetworkBase`. If `None`,
            a network is created as specified in :attr:`hparams`.
        network_kwargs (dict, optional): Keyword arguments for network
            constructor.
            Note that the `hparams` argument for network
            constructor is specified in the "network_hparams" field of
            :attr:`hparams` and should not be included in `network_kwargs`.
            Ignored if :attr:`network` is given.
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparamerter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter sturcture and
            default values.
    """
    def __init__(self,
                 network=None,
                 network_kwargs=None,
                 hparams=None):
        ModuleBase.__init__(self, hparams=hparams)

        with tf.variable_scope(self.variable_scope):
            self._build_network(network, network_kwargs)

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        .. role:: python(code)
           :language: python

        .. code-block:: python

            {
                'network_type': 'FeedForwardNetwork',
                'network_hparams': {
                    'layers': [
                        {
                            'type': 'Dense',
                            'kwargs': {'units': 256, 'activation': 'relu'}
                        },
                        {
                            'type': 'Dense',
                            'kwargs': {'units': 256, 'activation': 'relu'}
                        },
                    ]
                },
                'distribution_kwargs': None,
                'name': 'policy_net',
            }

        Here:

        "network_type" : str or class or instance
            A network that takes in state and returns outputs for
            generating actions. This can be a class, its name or module path,
            or a class instance. Ignored if `network` is given to the
            constructor.

        "network_hparams" : dict
            Hyperparameters for the network. With the :attr:`network_kwargs`
            argument to the constructor, a network is created with
            :python:`network_class(**network_kwargs, hparams=network_hparams)`.

            For example, the default values creates a two-layer dense network.

        "distribution_kwargs" : dict, optional
            Keyword arguments for distribution constructor. A distribution
            would be created for action sampling.

        "name" : str
            Name of the policy.
        """
        return {
            'network_type': 'FeedForwardNetwork',
            'network_hparams': {
                'layers': [
                    {
                        'type': 'Dense',
                        'kwargs': {'units': 256, 'activation': 'relu'}
                    },
                    {
                        'type': 'Dense',
                        'kwargs': {'units': 256, 'activation': 'relu'}
                    },
                ]
            },
            'distribution_kwargs': None,
            'name': 'policy_net',
            '@no_typecheck': ['network_type', 'network_hparams']
        }

    def _build_network(self, network, kwargs):
        if network is not None:
            self._network = network
        else:
            kwargs = utils.get_instance_kwargs(
                kwargs, self._hparams.network_hparams)
            self._network = utils.check_or_get_instance(
                self._hparams.network_type,
                kwargs,
                module_paths=['texar.modules', 'texar.custom'])

    def _build(self, inputs, mode=None): # pylint: disable=arguments-differ
        raise NotImplementedError

    @property
    def network(self):
        """The network.
        """
        return self._network


#TODO(zhiting): Allow structured discrete actions.
class CategoricalPolicyNet(PolicyNetBase):
    """Policy net with Categorical distribution for discrete scalar actions.

    This is a combination of a network with a top-layer distribution for
    action sampling.

    Args:
        action_space (optional): An instance of :class:`~texar.agents.Space`
            specifying the action space. If not given, an discrete action space
            `[0, high]` is created with `high` specified in :attr:`hparams`.
        network (optional): A network that takes in state and returns
            outputs for generating actions. For example, an instance of subclass
            of :class:`~texar.modules.FeedForwardNetworkBase`. If `None`,
            a network is created as specified in :attr:`hparams`.
        network_kwargs (dict, optional): Keyword arguments for network
            constructor.
            Note that the `hparams` argument for network
            constructor is specified in the "network_hparams" field of
            :attr:`hparams` and should not be included in `network_kwargs`.
            Ignored if :attr:`network` is given.
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparamerter will be set to default values. See
            :meth:`default_hparams` for the hyperparameter sturcture and
            default values.

    .. document private functions
    .. automethod:: _build
    """

    def __init__(self,
                 action_space=None,
                 network=None,
                 network_kwargs=None,
                 hparams=None):
        PolicyNetBase.__init__(self, hparams=hparams)

        with tf.variable_scope(self.variable_scope):
            if action_space is None:
                action_space = Space(
                    low=0, high=self._hparams.action_space, dtype=np.int32)
            self._action_space = action_space
            self._append_output_layer()

    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        .. code-block:: python

            {
                'network_type': 'FeedForwardNetwork',
                'network_hparams': {
                    'layers': [
                        {
                            'type': 'Dense',
                            'kwargs': {'units': 256, 'activation': 'relu'}
                        },
                        {
                            'type': 'Dense',
                            'kwargs': {'units': 256, 'activation': 'relu'}
                        },
                    ]
                },
                'distribution_kwargs': {
                    'dtype': 'int32',
                    'validate_args': False,
                    'allow_nan_stats': True
                },
                'action_space': 2,
                'make_output_layer': True,
                'name': 'categorical_policy_net'
            }

        Here:

        "distribution_kwargs" : dict
            Keyword arguments for the :tf_main:`Categorical
            <distributions/Categorical>` distribution constructor. Arguments
            `logits` and `probs` should not be included as they are inferred
            from the inputs. Argument `dtype` can be a string (e.g., `int32`)
            and will be converted to a corresponding tf dtype.

        "action_space" : int
            Upper bound of the action space. The resulting action space is
            all discrete scalar numbers between 0 and the upper bound specified
            here (both inclusive).

        "make_output_layer" : bool
            Whether to append a dense layer to the network to transform
            features to logits for action sampling. If `False`, the final layer
            output of network must match the action space.

        See :class:`~texar.modules.PolicyNetBase.default_hparams` for details
        of other hyperparameters.
        """
        hparams = PolicyNetBase.default_hparams()
        hparams.update({
            'distribution_kwargs': {
                'dtype': 'int32',
                'validate_args': False,
                'allow_nan_stats': True
            },
            'action_space': 2,
            'make_output_layer': True,
            'name': 'categorical_policy_net'
        })
        return hparams

    def _append_output_layer(self):
        if not self._hparams.make_output_layer:
            return

        if self._action_space.shape != ():
            raise ValueError('Only scalar discrete action is supported.')
        else:
            output_size = self._action_space.high - self._action_space.low

        layer_hparams = {
            'type': 'Dense',
            'kwargs': {'units': output_size}
        }
        self._network.append_layer(layer_hparams)

    def _build(self, inputs, mode=None):
        """Takes in states and outputs actions.

        Args:
            inputs: Inputs to the policy network with the first dimension
                the batch dimension.
            mode (optional): A tensor taking value in
                :tf_main:`tf.estimator.ModeKeys <estimator/ModeKeys>`, including
                `TRAIN`, `EVAL`, and `PREDICT`. If `None`,
                :func:`texar.global_mode` is used.

        Returns
            A `dict` including fields `"logits"`, `"action"`, and `"dist"`,
            where

            - **"logits"**: A Tensor of shape \
            `[batch_size] + action_space size` used for categorical \
            distribution sampling.
            - **"action"**: A Tensor of shape \
            `[batch_size] + action_space.shape`.
            - **"dist"**: The \
            :tf_main:`Categorical <distributions/Categorical>` based on the \
            logits.
        """
        logits = self._network(inputs, mode=mode)

        dkwargs = self._hparams.distribution_kwargs.todict()
        dkwargs['dtype'] = get_tf_dtype(dkwargs['dtype'])
        dist = tf.distributions.Categorical(logits=logits, **dkwargs)

        action = dist.sample()
        to_shape = [-1] # for batch dimension
        to_shape.extend(list(self._action_space.shape))
        action = tf.reshape(action, to_shape)

        outputs = {
            "logits": logits,
            "action": action,
            "dist": dist
        }

        if not self._built:
            self._add_internal_trainable_variables()
            self._add_trainable_variable(self._network.trainable_variables)
            self._built = True

        return outputs

    @property
    def action_space(self):
        """An instance of :class:`~texar.agents.Space` specifiying the
        action space.
        """
        return self._action_space
