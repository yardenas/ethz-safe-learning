from gym.envs.registration import register

register(
    id='ant-simba-v0',
    entry_point='simba.envs.ant:AntEnv',
    max_episode_steps=1000,
)

from simba.envs.ant.ant import AntEnv
