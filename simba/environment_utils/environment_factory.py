import gym
import safety_gym
import simba.envs


def make_environment(config):
    environment = config['options']['environment']
    return gym.make(environment)
