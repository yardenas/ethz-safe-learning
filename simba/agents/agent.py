from tqdm import tqdm
import simba.infrastructure.replay_buffer as rb
from simba.infrastructure.logging_utils import logger


class BaseAgent(object):
    """
    A base class for RL agents. An RL agent inherits from this
    class and implements a concrete RL algorithm.
    """

    def __init__(self,
                 replay_buffer_size,
                 add_observation_noise,
                 action_repeat,
                 *args,
                 **kwargs
                 ):
        self.replay_buffer = rb.ReplayBuffer(replay_buffer_size, add_observation_noise)
        self.action_repeat = action_repeat
        assert self.action_repeat, "Action repeat should be at least 1."
        self.training_report = dict()
        self.total_training_steps = 0

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
        steps = 0
        pbar = tqdm(total=max_trajectory_length)
        while steps < max_trajectory_length:
            action = policy.generate_action(observation)
            for i in range(self.action_repeat):
                images.append(environment.render(mode='rgb_array'))
                observation, _, done, _ = environment.step(action)
                steps += 1
                pbar.update(1)
                if done or steps == max_trajectory_length:
                    break
        pbar.close()
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
            trajectory, trajectory_length = self.sample_trajectory(
                environment,
                policy,
                max_trajectory_length,
                pbar)
            trajectories.append(trajectory)
            timesteps_this_batch += trajectory_length
        pbar.close()
        return trajectories, timesteps_this_batch

    def sample_trajectory(
            self,
            environment,
            policy,
            max_trajectory_length,
            pbar):
        observation = environment.reset()
        observations, actions, rewards, next_observations, terminals, infos = \
            [], [], [], [], [], []
        steps = 0
        rollout_done = False
        while not rollout_done:
            action = policy.generate_action(observation)
            observations.append(observation)
            actions.append(action)
            repeat_rewards = 0.0
            for _ in range(self.action_repeat):
                observation, reward, done, info = \
                    environment.step(action)
                steps += 1
                repeat_rewards += reward
                rollout_done = (steps == max_trajectory_length) or done
                if info.get('goal_met', False):
                    print("hellow")
                    print(observation[5] - observations[-1][5])
                    break
                pbar.update(1)
                if rollout_done:
                    break
            next_observations.append(observation)
            rewards.append(repeat_rewards)
            infos.append(info)
            terminals.append(rollout_done)
        # A more safe assert would be to check all of the actions, but compute time is not cheap.
        # (Although premature optimization is the root of all evil.)
        assert actions[0].shape == environment.action_space.shape, "Policy produces wrong actions shape."
        return rb.path_summary(
            observations,
            actions,
            rewards,
            next_observations,
            terminals,
            infos), steps
