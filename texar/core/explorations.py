#
"""
TODO: docs
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from texar.hyperparams import HParams

# pylint: disable=invalid-name

class ExplorationBase(object):
    """Base class inherited by all exploration classes.

    Args:

    """
    def __init__(self, hparams=None):
        self._hparams = HParams(hparams, self.default_hparams())

    @staticmethod
    def default_hparams():
        """Returns a dictionary of default hyperparameters.

        TODO: docs
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
    """TODO: docs

    Args:
    """
    def __init__(self, hparams=None):
        ExplorationBase.__init__(self, hparams=hparams)

    @staticmethod
    def default_hparams():
        """TODO
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

