import gym
import safety_gym


def make_environment(config):
    environment = gym.make(config['options']['environment'])
    environment.seed(config['options']['seed'])
    return environment
