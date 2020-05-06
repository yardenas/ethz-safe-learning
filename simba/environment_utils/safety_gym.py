import gym
import safety_gym
from simba.environment_utils.safety_gym_scoring import SafetyGymStateScorer
from simba.environment_utils.mbrl_env import MbrlEnv


class MbrlSafetyGymEnv(MbrlEnv, gym.Wrapper):
    def __init__(self, task_name):
        environment = gym.make(task_name)
        super().__init__(env=environment)
        self._scorer = SafetyGymStateScorer(environment.config)

    def get_reward(self, obs, acs):
        return self._scorer.reward(obs, acs)
