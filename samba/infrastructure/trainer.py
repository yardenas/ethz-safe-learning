from samba.agents.agent import BaseAgent
from samba.infrastructure.logger import Logger

# TODO (yarden): Take parameters explicitly and not as a dictionary of params.
class RLTrainer(object):
    def __init__(self, agent, environemnt, params):
        self.params = params
        self.logger = Logger(self.params['logdir'])
        self.agent = agent
        self.environment = environemnt

    def train(self):
        for epoch in range(self.params['epochs']):
            self.agent.interact(self.environment)
            self.agent.update_model()
            self.agent.update_policy()
            if epoch % self.params['log_frequency'] == 0:
                self.log(self.agent.report())
            if epoch % self.params['video_log_frequency'] == 0:
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
        self.logger.log_paths_as_videos(
            trajectory_records, epoch, self.params['fps'],
            len(trajectory_records), self.params['max_video_length']
        )

