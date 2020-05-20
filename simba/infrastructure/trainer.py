import numpy as np

from simba.infrastructure.logging_utils import TrainingLogger
from simba.infrastructure.logging_utils import logger


class RLTrainer(object):
    def __init__(self,
                 agent,
                 environemnt,
                 log_frequency,
                 video_log_frequency,
                 max_video_length,
                 eval_interaction_steps,
                 eval_episode_length,
                 training_logger_params):
        self.agent = agent
        self.environment = environemnt
        self.training_logger = TrainingLogger(**training_logger_params)
        self.log_frequency = log_frequency
        self.max_video_length = max_video_length
        self.video_log_frequency = video_log_frequency
        self.eval_interaction_steps = eval_interaction_steps
        self.eval_episode_length = eval_episode_length

    def train(self, iterations):
        self.agent.build_graph()
        for iteration in range(iterations):
            logger.info("Training iteration {}.".format(iteration))
            self.agent.interact(self.environment)
            self.agent.update()
            if self.log_frequency > 0 and iteration % self.log_frequency == 0:
                self.log(self.agent.report(
                    self.environment,
                    self.eval_interaction_steps,
                    self.eval_episode_length
                ), iteration)
            if self.video_log_frequency > 0 and iteration % self.video_log_frequency == 0:
                self.log_video(self.agent.render_trajectory(
                    environment=self.environment,
                    policy=self.agent.policy,
                    max_trajectory_length=self.max_video_length
                ), iteration)

    def play_trained_model(self):
        # self.agent.load_graph()
        pass

    def log(self, report, epoch):
        """
        Takes a report from the agent and logs it.
        """
        train_return_values = np.array([trajectory['reward'].sum()
                                        for trajectory in report.pop('training_trajectories')])
        report.update(dict(
            training_rl_objective=train_return_values.mean(),
            sum_rewards_stddev=train_return_values.std()
        ))
        # losses = report.pop('losses')
        # for i, loss in enumerate(losses):
        #     self.training_logger.log_scalars(
        #         scalar_dict={'loss': loss},
        #         group_name='losses/' + str(epoch),
        #         step=i,
        #     )

        predicted_trajectory, ground_truth_trajectory = report.pop('predicted_states_vs_ground_truth')
        for i, (predicted_state, ground_truth_state) in \
                enumerate(zip(predicted_trajectory.transpose(), ground_truth_trajectory.transpose())):
            for t, (predicted_value, ground_truth_value) in enumerate(zip(predicted_state, ground_truth_state)):
                self.training_logger.log_scalars(
                    scalar_dict={'ground truth': ground_truth_value,
                                 'predicted': predicted_value.mean()},
                    group_name='states/epoch/' + str(epoch) + '/state_id/' + str(i),
                    step=t
                )
        for key, value in report.items():
            self.training_logger.log_scalar(value, key, epoch)
        self.training_logger.flush()

    def log_video(self, trajectory_records, epoch):
        """
        Logs videos from rendered trajectories.
        """
        trajectory_rendering = trajectory_records
        video = np.transpose(trajectory_rendering, [0, 3, 1, 2])
        self.training_logger.log_video(
            np.expand_dims(video, axis=0),
            'what_the_policy_looks_like',
            epoch)
