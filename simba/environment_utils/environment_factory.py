import gym
import safety_gym


def make_environment(config):
    environment = config['options']['environment']
    return gym.make(environment)
