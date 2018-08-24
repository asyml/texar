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
Classes and utilities for exploration in RL.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from texar.hyperparams import HParams

# pylint: disable=invalid-name

__all__ = [
    "ExplorationBase",
    "EpsilonLinearDecayExploration"
]

class ExplorationBase(object):
    """Base class inherited by all exploration classes.

    Args:
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparameters are set to default values. See
            :meth:`default_hparams` for the defaults.
    """
    def __init__(self, hparams=None):
        self._hparams = HParams(hparams, self.default_hparams())

    @staticmethod
    def default_hparams():
        """Returns a `dict` of hyperparameters and their default values.

        .. code-block:: python

            {
                'name': 'exploration_base'
            }
        """
        return {
            'name': 'exploration_base'
        }

    def get_epsilon(self, timestep):
        """Returns the epsilon value.

        Args:
            timestep (int): The time step.

        Returns:
            float: the epsilon value.
        """
        raise NotImplementedError

    @property
    def hparams(self):
        """The hyperparameter.
        """
        return self._hparams


class EpsilonLinearDecayExploration(ExplorationBase):
    """Decays epsilon linearly.

    Args:
        hparams (dict or HParams, optional): Hyperparameters. Missing
            hyperparameters are set to default values. See
            :meth:`default_hparams` for the defaults.
    """
    def __init__(self, hparams=None):
        ExplorationBase.__init__(self, hparams=hparams)

    @staticmethod
    def default_hparams():
        """Returns a `dict` of hyperparameters and their default values.

        .. code-block:: python

            {
                'initial_epsilon': 0.1,
                'final_epsilon': 0.0,
                'decay_timesteps': 20000,
                'start_timestep': 0,
                'name': 'epsilon_linear_decay_exploration',
            }

        This specifies the decay process that starts at
        "start_timestep" with the value "initial_epsilon", and decays for
        steps "decay_timesteps" to reach the final epsilon value
        "final_epsilon".
        """
        return {
            'name': 'epsilon_linear_decay_exploration',
            'initial_epsilon': 0.1,
            'final_epsilon': 0.0,
            'decay_timesteps': 20000,
            'start_timestep': 0
        }

    def get_epsilon(self, timestep):
        nsteps = self._hparams.decay_timesteps
        st = self._hparams.start_timestep
        et = st + nsteps

        if timestep <= st:
            return self._hparams.initial_epsilon
        if timestep > et:
            return self._hparams.final_epsilon
        r = (timestep - st) * 1.0 / nsteps
        epsilon = (1 - r) * self._hparams.initial_epsilon + \
                r * self._hparams.final_epsilon

        return epsilon

