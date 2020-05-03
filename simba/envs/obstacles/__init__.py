from gym.envs.registration import register

register(
    id='obstacles-simba-v0',
    entry_point='simba.envs.obstacles:Obstacles',
    max_episode_steps=500,
)
from simba.envs.obstacles.obstacles_env import Obstacles
