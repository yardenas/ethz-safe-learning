from gym.envs.registration import register

register(
    id='Reacher7DOF-v0',
    entry_point='simba.environment_utils.reacher_7dof.reacher_7dof:Reacher7DOFEnv',
    max_episode_steps=500,
)
