from txtgen.hyperparams import HParams


class AgentBase:
    def __init__(self, hparams=None):
        self._hparams = HParams(hparams, self.default_hparams())
        self.current_state = None
        self.timestep = 0

    @staticmethod
    def default_hparams():
        return {
            'name': 'agent_base'
        }

    def set_initial_state(self, observation):
        raise NotImplementedError

    def perceive(self, action, reward, is_terminal, next_state):
        raise NotImplementedError

    def get_action(self, state, action_mask=None):
        raise NotImplementedError
