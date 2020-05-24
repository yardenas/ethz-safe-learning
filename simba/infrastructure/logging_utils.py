import os
from tensorboardX import SummaryWriter
import numpy as np
import logging

logger = logging.getLogger('simba')


def init_loggging(log_level='WARNING', log_file=None):
    logging.basicConfig(
        format='%(asctime)s:%(name)s:%(levelname)s: %(message)s',
        level=log_level,
        filename=log_file
    )
    logger.setLevel(log_level)


class TrainingLogger:
    """
    Copy-pasted from 'Berkely CS285'
    (https://github.com/yardenas/berkeley-deep-rl/tree/f741338c085ee5b329f3c9dd05e93e89bc43574a)
    and used for dumping statistics to to TensorBoard readable file.
    """
    def __init__(self,
                 log_dir,
                 fps):
        self._log_dir = log_dir
        self.fps = fps
        logger.info('Logging training data to: ' + log_dir)
        self._summ_writer = SummaryWriter(log_dir, flush_secs=1, max_queue=1)

    def log_scalar(self, scalar, name, step):
        self._summ_writer.add_scalar('{}'.format(name), scalar, step)

    def log_scalars(self, scalar_dict, group_name, step):
        """Will log all scalars in the same plot."""
        self._summ_writer.add_scalars('{}'.format(group_name), scalar_dict, step)

    def log_image(self, image, name, step):
        assert(len(image.shape) == 3)  # [C, H, W]
        self._summ_writer.add_image('{}'.format(name), image, step)

    def log_video(self, video_frames, name, step):
        assert len(video_frames.shape) == 5, "Need [N, T, C, H, W] input tensor for video logging!"
        self._summ_writer.add_video('{}'.format(name), video_frames, step, fps=self.fps)

    def log_figures(self, figure, name, step):
        """figure: matplotlib.pyplot figure handle"""
        assert figure.shape[0] > 0, "Figure logging requires input shape [batch x figures]!"
        self._summ_writer.add_figure('{}'.format(name), figure, step)

    def log_figure(self, figure, name, step):
        """figure: matplotlib.pyplot figure handle"""
        self._summ_writer.add_figure('{}'.format(name), figure, step)

    def log_graph(self, graph, step, phase):
        """figure: matplotlib.pyplot figure handle"""
        self._summ_writer.add_graph(graph)

    def log_histogram(self, data, name, step):
        self._summ_writer.add_histogram(name, data, step)

    def dump_scalars(self, log_path=None):
        log_path = os.path.join(self._log_dir, "scalar_data.json") if log_path is None else log_path
        self._summ_writer.export_scalars_to_json(log_path)

    def flush(self):
        self._summ_writer.flush()
