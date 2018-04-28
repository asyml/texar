#
"""
Base class for reinforcement learning agents for sequence prediction.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from texar.agents.agent_base import AgentBase

# pylint: disable=too-many-instance-attributes

class SeqAgentBase(AgentBase):
    """
    Base class inherited by sequence prediction RL agents.

    Args:
        TODO
    """
    def __init__(self, hparams=None):
        AgentBase.__init__(self, hparams)


    @staticmethod
    def default_hparams():
        """Returns a dictionary of hyperparameters with default values.

        TODO
        """
        return {
            'name': 'agent'
        }

