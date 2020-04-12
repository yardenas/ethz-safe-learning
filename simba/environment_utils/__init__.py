from gym.envs.registration import register
from simba.environment_utils.safety_gym_scoring import SafetyGymStateScorer

register(
    id='MBRLCartpole-v0',
    entry_point='simba.environment_utils.cartpole:CartpoleEnv'
)
