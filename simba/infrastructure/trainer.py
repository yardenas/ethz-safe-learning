from simba.agents.agent import BaseAgent
from simba.infrastructure.logger import Logger


class RLTrainer(object):
    def __init__(self,
                 agent,
                 environemnt,
                 seed,
                 epochs,
                 logger_kwargs,
                 log_frequency,
                 video_log_frequency):
        self.agent = agent
        self.environment = environemnt
        self.seed = seed
        self.epochs = epochs
        self.logger = Logger(**logger_kwargs)
        self.log_frequency = log_frequency
        self.video_log_frequency = video_log_frequency

    def train(self):
        for epoch in range(self.epochs):
            print("Training epoch {}.".format(epoch))
            self.agent.interact(self.environment)
            self.agent.update_model()
            self.agent.update_policy()
            if epoch % self.log_frequency == 0:
                self.log(self.agent.report())
            if epoch % self.video_log_frequency == 0:
                self.log_video(self.agent.say_cheese())

    def play_trained_model(self):
        pass

    def log(self, report, epoch):
        """
        Takes a report from the agent and logs it.
        """
        for key, value in report.items():
            print('{} : {}'.format(key, value))
            self.logger.log_scalar(value, key, epoch)
        self.logger.flush()

    def log_video(self, trajectory_records, epoch):
        """
        Logs videos from rendered trajectories.
        """
        self.logger.log_paths_as_videos(
            trajectory_records, epoch,
            len(trajectory_records)
        )

