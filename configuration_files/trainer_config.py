from configuration_files.agent_config import mbrl_agent_cfg_dict
from simba.agents import MBRLAgent

trainer_cfg_dict = {
    'agent': MBRLAgent(**mbrl_agent_cfg_dict),
    'environment': None,
    'seed': 42,
    'log_frequency': 1,
    'video_log_frequency': 0,
    'training_logger_kwargs': {
        'log_dir': '/data',
        'fps': 10,
        'max_video_samples': 10
    }
}
