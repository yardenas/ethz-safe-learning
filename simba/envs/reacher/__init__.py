from gym.envs.registration import register

register(
    id='reacher-simba-v0',
    entry_point='simba.envs.reacher:Reacher7DOFEnv',
    max_episode_steps=500,
)

from simba.envs.reacher.reacher_env import Reacher7DOFEnv
