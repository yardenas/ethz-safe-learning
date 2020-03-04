import numpy as np


class ReplayBuffer(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.paths = []
        self.obs = None
        self.acs = None
        self.concatenated_rews = None
        self.next_obs = None
        self.terminals = None
        self.add_noise = None

    def add_rollouts(self, paths):
        # add new rollouts into our list of rollouts
        for path in paths:
            self.paths.append(path)
        # convert new rollouts into their component arrays, and append them onto our arrays
        observations, actions, next_observations,\
            terminals, concatenated_rews = \
            concatenate_rollouts(paths)
        if self.add_noise:
            observations = add_noise(observations)
            next_observations = add_noise(next_observations)

        if self.obs is None:
            self.obs = observations[-self.max_size:]
            self.acs = actions[-self.max_size:]
            self.next_obs = next_observations[-self.max_size:]
            self.terminals = terminals[-self.max_size:]
            self.concatenated_rews = \
                concatenated_rews[-self.max_size:]
        else:
            self.obs = np.concatenate(
                [self.obs, observations])[-self.max_size:]
            self.acs = np.concatenate(
                [self.acs, actions])[-self.max_size:]
            self.next_obs = np.concatenate(
                [self.next_obs, next_observations])[-self.max_size:]
            self.terminals = np.concatenate(
                [self.terminals, terminals])[-self.max_size:]
            self.concatenated_rews = np.concatenate(
                [self.concatenated_rews, concatenated_rews])[-self.max_size:]

    def sample_random_rollouts(self, num_rollouts):
        rand_indices = np.random.permutation(len(self.paths))[:num_rollouts]
        return self.paths[rand_indices]

    def sample_recent_rollouts(self, num_rollouts=1):
        return self.paths[-num_rollouts:]

    def sample_random_data(self, batch_size):
        assert self.obs.shape[0] == self.acs.shape[0] == \
               self.concatenated_rews.shape[0] == self.next_obs.shape[0] == self.terminals.shape[0]
        rand_indices = np.random.permutation(self.obs.shape[0])[:batch_size]
        return self.obs[rand_indices],\
            self.acs[rand_indices],\
            self.concatenated_rews[rand_indices],\
            self.next_obs[rand_indices],\
            self.terminals[rand_indices]

    def sample_recent_data(self, batch_size=1, concat_rew=True):
        if concat_rew:
            return self.obs[-batch_size:], self.acs[-batch_size:], self.concatenated_rews[-batch_size:], \
                   self.next_obs[-batch_size:], self.terminals[-batch_size:]
        else:
            num_recent_rollouts_to_return = 0
            num_datapoints_so_far = 0
            index = -1
            while num_datapoints_so_far < batch_size:
                recent_rollout = self.paths[index]
                index -= 1
                num_recent_rollouts_to_return += 1
                num_datapoints_so_far += path_length(recent_rollout)
            rollouts_to_return = self.paths[-num_recent_rollouts_to_return:]
            observations, actions, next_observations, terminals, concatenated_rews = \
                concatenate_rollouts(rollouts_to_return)
            return observations, actions, next_observations, terminals


def concatenate_rollouts(paths):
    """
    Take a list of rollout dictionaries
    and return separate arrays,
    where each array is a concatenation of that array from across the rollouts
    """
    observations = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    next_observations = np.concatenate([path["next_observation"] for path in paths])
    terminals = np.concatenate([path["terminal"] for path in paths])
    concatenated_rewards = np.concatenate([path["reward"] for path in paths])
    return observations, actions, next_observations, terminals, concatenated_rewards


def add_noise(data, noise_to_signal):
    """
    Noise data with a normal distribution with variance that's
    proportional to mean of each dimension.
    :param data:
    :param noise_to_signal:
    :return:
    """
    mean_data = np.mean(data, axis=0)
    mean_data[mean_data == 0] = 1e-8
    std_of_noise = np.abs(mean_data * noise_to_signal)
    return data + np.random.normal(0.0, std_of_noise, data.shape)


def path_length(path):
    """
    :param path:
    :return:
    """
    return len(path['reward'])
