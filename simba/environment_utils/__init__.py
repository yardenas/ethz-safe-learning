from gym.envs.registration import register
from simba.environment_utils.safety_gym_scoring import SafetyGymStateScorer

register(
    id='MbrlInvertedPendulum-v2',
    entry_point='simba.environment_utils.inverted_pendulum:MbrlInvertedPendulumEnv'
)
