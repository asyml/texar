from texar.hyperparams import HParams


class ExplorationBase:
    def __init__(self, hparams=None):
        self._hparams = HParams(hparams, self.default_hparams())

    @staticmethod
    def default_hparams():
        return {
            'name': 'exploration_base'
        }

    @property
    def epsilon(self):
        raise NotImplementedError

    def add_timestep(self):
        raise NotImplementedError


class EpsilonDecayExploration(ExplorationBase):
    def __init__(self, hparams=None):
        ExplorationBase.__init__(self, hparams=hparams)
        self._epsilon = self._hparams.initial_epsilon
        self.timestep = 0
        self.initial_epsilon = self._hparams.initial_epsilon
        self.final_epsilon = self._hparams.final_epsilon
        self.decay_steps = self._hparams.decay_steps

    @staticmethod
    def default_hparams():
        return {
            'name': 'epsilon_decay_exploration',
            'initial_epsilon': 0.1,
            'final_epsilon': 0.0,
            'decay_steps': 20000
        }

    def epsilon(self):
        return self._epsilon

    def add_timestep(self):
        self.timestep += 1
        self.epsilon_decay()

    def epsilon_decay(self):
        if self._epsilon > 0.:
            self._epsilon -= (self.initial_epsilon - self.final_epsilon) / self.decay_steps
