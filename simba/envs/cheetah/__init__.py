from gym.envs.registration import register

register(
    id='cheetah-simba-v0',
    entry_point='simba.envs.cheetah:HalfCheetahEnv',
    max_episode_steps=1000,
)
from simba.envs.cheetah.cheetah import HalfCheetahEnv
