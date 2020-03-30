
agent_cfg_dict = {
    'train_batch_size': 100,
    'train_interaction_steps': 100,
    'eval_interaction_steps': 100,
    'episode_length': 100,
    'replay_buffer_size': 1e6
}

mbrl_agent_cfg_dict = {
    'warmup_policy': None,
    'warmup_timesteps': 1,
    'policy': None,
    'model': None,
}.update(agent_cfg_dict)
