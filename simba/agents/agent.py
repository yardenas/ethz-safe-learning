from tqdm import tqdm
import simba.infrastructure.replay_buffer as rb
from simba.infrastructure.logging_utils import logger


class BaseAgent(object):
    """
    A base class for RL agents. An RL agent inherits from this
    class and implements a concrete RL algorithm.
    """

    def __init__(self,
                 seed,
                 replay_buffer_size,
                 add_observation_noise,
                 action_repeat,
                 *args,
                 **kwargs
                 ):
        self.replay_buffer = rb.ReplayBuffer(replay_buffer_size, add_observation_noise)
        self.action_repeat = action_repeat
        self.set_random_seeds(seed)
        self.training_report = dict()
        self.total_training_steps = 0

    def set_random_seeds(self, seed):
        raise NotImplementedError("Random seeds function must be implemented.")

    def interact(self, environment):
        samples, timesteps_this_batch = self._interact(environment)
        self.total_training_steps += timesteps_this_batch
        self.replay_buffer.store(samples)
        self.training_report.update(dict(
            training_trajectories=samples,
            total_training_steps=self.total_training_steps
        ))

    def update(self):
        raise NotImplementedError

    def _interact(self, environment):
        raise NotImplementedError

    def build_graph(self, graph_dir=None):
        if graph_dir is None:
            logger.info("Building computational graph.")
            self._build()
        else:
            logger.info("Loading computational graph from {}".format(graph_dir))
            self._load()

    def _build(self):
        raise NotImplementedError

    def _load(self):
        raise NotImplementedError

    def report(self,
               environment,
               eval_interaction_steps,
               eval_episode_length):
        return self.training_report

    def render_trajectory(
            self,
            environment,
            policy,
            max_trajectory_length):
        observation = environment.reset()
        images = []
        logger.info("Sampling render trajectory.")
        for _ in tqdm(range(max_trajectory_length)):
            images.append(environment.render(mode='rgb_array'))
            action = policy.generate_action(observation)
            observation, _, done, _ = environment.step(action)
            if done:
                break
        return images

    def sample_trajectories(
            self,
            environment,
            policy,
            batch_size,
            max_trajectory_length):
        timesteps_this_batch = 0
        trajectories = []
        pbar = tqdm(total=batch_size)
        while timesteps_this_batch < batch_size:
            trajectories.append(self.sample_trajectory(
                environment,
                policy,
                max_trajectory_length,
                pbar))
            timesteps_this_batch += rb.path_length(trajectories[-1])
        pbar.close()
        return trajectories, timesteps_this_batch

    def sample_trajectory(
            self,
            environment,
            policy,
            max_trajectory_length,
            pbar):
        observation = environment.reset()
        observations, actions, \
        rewards, next_observations, \
        terminals, image_obs = [], [], [], [], [], []
        steps = 0
        rollout_done = False
        while not rollout_done:
            action = policy.generate_action(observation)
            for _ in range(self.action_repeat):
                observations.append(observation)
                actions.append(action)
                observation, reward, done, info = \
                    environment.step(action)
                steps += 1
                next_observations.append(observation)
                rewards.append(reward)
                rollout_done = (steps == max_trajectory_length) or done
                terminals.append(rollout_done)
                if info.get('goal_met', False):
                    [data.pop() for data in [observations, actions, next_observations, rewards, terminals]]
                pbar.update(1)
                if rollout_done:
                    break
        # A more safe assert would be to check all of the actions, but compute time is not cheap.
        # (Although premature optimization is the root of all evil.)
        assert actions[0].shape == environment.action_space.shape, "Policy produces wrong actions shape."
        return rb.path_summary(
            observations,
            actions,
            rewards,
            next_observations,
            terminals
        )
