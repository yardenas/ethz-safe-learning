from simba.infrastructure.training_logger import TrainingLogger
from simba.infrastructure.logger import logger
# TODO (yarden): set logging level from argparse on outer levels...


class RLTrainer(object):
    def __init__(self,
                 agent,
                 environemnt,
                 seed,
                 log_frequency,
                 video_log_frequency,
                 training_logger_kwargs):
        self.agent = agent
        self.environment = environemnt
        self.seed = seed
        self.training_logger = TrainingLogger(**training_logger_kwargs)
        self.log_frequency = log_frequency
        self.video_log_frequency = video_log_frequency

    def train(self):
        self.agent.build_graph()
        converged = False
        iteration = 0
        while not converged:
            logger.info("Training iteration {}.".format(iteration))
            self.agent.interact(self.environment)
            self.agent.update()
            if iteration % self.log_frequency == 0:
                self.log(self.agent.report(), iteration)
            if iteration % self.video_log_frequency == 0:
                self.log_video(self.agent.say_cheese(), iteration)
            iteration += 1

    def play_trained_model(self):
        # self.agent.load_graph()
        pass

    def log(self, report, epoch):
        """
        Takes a report from the agent and logs it.
        """
        for key, value in report.items():
            print('{} : {}'.format(key, value))
            self.training_logger.log_scalar(value, key, epoch)
        self.training_logger.flush()

    def log_video(self, trajectory_records, epoch):
        """
        Logs videos from rendered trajectories.
        """
        self.training_logger.log_paths_as_videos(
            trajectory_records, epoch,
            len(trajectory_records)
        )

