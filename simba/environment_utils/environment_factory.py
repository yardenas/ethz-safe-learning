import gym
import safety_gym
import simba.environment_utils as env_utils


def make_environment(config):
    environment_name = config['options']['environment']
    # Mbrl envs using the safety-gym suite need to specify a task name to make a concrete environment.
    if environment_name.startswith('MbrlSafe'):
        environment = env_utils.MbrlSafetyGym(environment_name.replace('Mbrl', ''))
    # Mbrl envs are wrapping their respective openai-gym envs, so the constructor api is slightly different.
    elif environment_name.startswith('Mbrl'):
        environment = eval('env_utils.' + environment_name)()
    else:
        environment = gym.make(environment_name)
    environment.seed(config['options']['seed'])
    return environment
