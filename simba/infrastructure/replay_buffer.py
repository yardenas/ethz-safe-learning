import numpy as np


class ReplayBuffer(object):
    def __init__(self,
                 max_size,
                 add_noise):
        self.max_size = max_size
        self.paths = []
        self.observations = None
        self.actions = None
        self.rewards = None
        self.next_observations = None
        self.terminals = None
        self.infos = None
        self.add_noise = add_noise

    def store(self, paths):
        # add new rollouts into our list of rollouts
        for path in paths:
            self.paths.append(path)
        # convert new rollouts into their component arrays, and append them onto our arrays
        observations, actions, next_observations, terminals, rewards, infos = \
            concatenate_rollouts(paths)
        if self.add_noise:
            observations = add_noise(observations)
            next_observations = add_noise(next_observations)
        if self.observations is None:
            self.observations = observations[-self.max_size:]
            self.actions = actions[-self.max_size:]
            self.next_observations = next_observations[-self.max_size:]
            self.terminals = terminals[-self.max_size:]
            self.rewards = rewards[-self.max_size:]
            self.infos = infos[-self.max_size:]
        else:
            self.observations = np.concatenate(
                [self.observations, observations])[-self.max_size:]
            self.actions = np.concatenate(
                [self.actions, actions])[-self.max_size:]
            self.next_observations = np.concatenate(
                [self.next_observations, next_observations])[-self.max_size:]
            self.terminals = np.concatenate(
                [self.terminals, terminals])[-self.max_size:]
            self.rewards = np.concatenate(
                [self.rewards, rewards])[-self.max_size:]
            self.infos = np.concatenate(
                [self.infos, infos])[-self.max_size:]

    def sample_random_rollouts(self, num_rollouts):
        rand_indices = np.random.permutation(len(self.paths))[:num_rollouts]
        return concatenate_rollouts(np.array(self.paths, copy=False)[rand_indices])

    def sample_recent_rollouts(self, num_rollouts=1):
        return self.paths[-num_rollouts:]

    def sample_random_data(self, batch_size):
        assert self.observations.shape[0] == self.actions.shape[0] == \
               self.rewards.shape[0] == self.next_observations.shape[0] == self.terminals.shape[0]
        rand_indices = np.random.permutation(self.observations.shape[0])[:batch_size]
        return self.observations[rand_indices], \
               self.actions[rand_indices], \
               self.next_observations[rand_indices], \
               self.terminals[rand_indices], \
               self.rewards[rand_indices], \
               self.infos[rand_indices]

    def sample_recent_data(self, batch_size):
        return self.observations[-batch_size:], self.actions[-batch_size:], self.next_observations[-batch_size:], \
               self.terminals[-batch_size:], self.rewards[-batch_size:], self.infos[-batch_size:]


def path_summary(observations,
                 actions,
                 rewards,
                 next_observations,
                 terminals,
                 infos):
    """
        Take info (separate arrays) from a single rollout
        and return it in a single dictionary
    """
    return {"observation": np.array(observations, dtype=np.float32),
            "reward": np.array(rewards, dtype=np.float32),
            "action": np.array(actions, dtype=np.float32),
            "next_observation": np.array(next_observations, dtype=np.float32),
            "terminal": np.array(terminals, dtype=np.float32),
            "info": infos}


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


def add_noise(data, noise_to_signal=0.01):
    """
    Noise data with a normal distribution with variance that's
    proportional to mean of each dimension.
    :param data:
    :param noise_to_signal:
    :return:
    """
    mean_data = np.mean(data, axis=0)
    mean_data[mean_data == 0] = 1e-5
    std_of_noise = np.abs(mean_data * noise_to_signal)
    return (data + np.random.normal(0.0, std_of_noise, data.shape)).astype(np.float32)


def path_length(path):
    """
    :param path:
    :return:
    """
    return len(path['reward'])
