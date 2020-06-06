import numpy as np
import tensorflow as tf
from simba.environment_utils.mbrl_env import MbrlEnv
from simba.infrastructure.logging_utils import logger


class MbrlSafetyGym(MbrlEnv):
    def __init__(self, task_name):
        super().__init__(task_name)
        offset = 0
        observation_space_summary = "Observation space indices are: "
        self.sensor_offset_table = dict()
        for k, value in sorted(self.env.obs_space_dict.items()):
            k_size = np.prod(value.shape)
            self.sensor_offset_table[k] = slice(offset, offset + k_size)
            observation_space_summary += \
                (k + " [" + str(offset) + ", " + str(offset + k_size) + ") " + "\n")
            offset += k_size
        logger.debug(observation_space_summary)
        self._scorer = SafetyGymStateScorer(
            self.env.config,
            self.sensor_offset_table)

    def get_reward(self, obs, acs, *args, **kwargs):
        return self._scorer.reward(obs, *args, **kwargs)

    def get_cost(self, obs, acs, *args, **kwargs):
        return self._scorer.cost(obs)

    def fix_observation(self, observation):
        # Predicting distances in exponential-space seems to really hold back the model from learning anything.
        observation[self.sensor_offset_table['goal_dist']] = -np.log(observation[self.sensor_offset_table['goal_dist']])
        observation[self.sensor_offset_table['accelerometer']][2] += np.random.normal(loc=0, scale=0.01)
        return observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.fix_observation(observation), reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset()
        return self.fix_observation(observation)


class SafetyGymStateScorer(object):
    def __init__(self, config, sensor_offset_table):
        for key, value in config.items():
            setattr(self, key, value)
        self.last_box_observed = False
        self.sensor_offset_table = sensor_offset_table

    def reward(self, observations, next_observations):
        reward = tf.zeros((tf.shape(observations)[0],))
        goal_achieved = tf.zeros_like(reward, dtype=tf.bool)
        # Distance from robot to goal
        if self.task == 'goal':
            dist_goal = self.goal_distance_metric(observations)
            next_dist_goal = self.goal_distance_metric(next_observations)
            goal_achieved = tf.less_equal(dist_goal, self.goal_size - 0.1)
            reward += (dist_goal - next_dist_goal) * self.reward_distance + tf.cast(goal_achieved,
                                                                                    tf.float32) * self.reward_goal
        # Distance from robot to box
        elif self.task == 'push':
            box_observed = tf.math.reduce_any(
                tf.greater(observations[:, self.sensor_offset_table['box_lidar']], 0.0), axis=1)
            next_box_observed = tf.math.reduce_any(
                tf.greater(next_observations[:, self.sensor_offset_table['box_lidar']], 0.0), axis=1)
            rewards_gate = tf.logical_and(box_observed, next_box_observed)
            dist_box_goal, dist_box = self.push_distance_metric(observations)
            next_dist_box_goal, next_dist_box = self.push_distance_metric(next_observations)
            goal_achieved = tf.less_equal(dist_box_goal, self.goal_size)
            reward += ((dist_box_goal - next_dist_box_goal) * self.reward_box_goal +
                       tf.cast(goal_achieved, tf.float32) * self.reward_goal) * tf.cast(rewards_gate, tf.float32)
            gate_dist_box_reward = tf.greater(dist_box, self.box_null_dist * self.box_size)
            reward += ((dist_box - next_dist_box) * self.reward_box_dist *
                       tf.cast(gate_dist_box_reward, tf.float32)) * tf.cast(rewards_gate, tf.float32)
        # Intrinsic reward for uprightness
        if self.reward_orientation:
            accelerometer = observations[:, self.sensor_offset_table['acceleration']]
            zalign = (accelerometer / tf.linalg.norm(accelerometer, axis=1, keepdims=True))
            reward += self.reward_orientation_scale * tf.linalg.tensordot(zalign, ([0.0, 0.0, 1.0]))
        # Clip reward
        if self.reward_clip:
            reward = tf.clip_by_value(reward, -self.reward_clip, self.reward_clip)
        return reward, goal_achieved

    def cost(self, observations):
        cost = tf.zeros((tf.shape(observations)[0],))
        # Conctacts processing
        if self.constrain_vases:
            vases_lidar = observations[:, self.sensor_offset_table['vases_lidar']]
            vases_dist = self.closest_distance(vases_lidar)
            cost += (vases_dist <= self.vases_size)
        if self.constrain_hazards:
            hazards_lidar = observations[:, self.sensor_offset_table['hazards_lidar']]
            hazards_dist = self.closest_distance(hazards_lidar)
            cost += hazards_dist <= self.hazards_size
            # print("fake hazards ", hazards_dist)
        if self.constrain_pillars:
            pillars_lidar = observations[:, self.sensor_offset_table['pillars_lidar']]
            pillars_dist = self.closest_distance(pillars_lidar)
            cost += (pillars_dist <= self.pillars_size)
        if self.constrain_gremlins:
            gremlins_lidar = observations[:, self.sensor_offset_table['gremlins_lidar']]
            gremlins_dist = self.closest_distance(gremlins_lidar)
            cost += (gremlins_dist <= self.gremlins_lidar)
        if self.constrain_indicator:
            return tf.greater(cost > 0.0)
        return cost

    def goal_distance_metric(self, observations):
        # Just a fancy way to clip negative values.
        return tf.squeeze(tf.nn.relu(observations[:, self.sensor_offset_table['goal_dist']]))

    def push_distance_metric(self, observations):
        box_lidar = observations[:, self.sensor_offset_table['box_lidar']]
        dist_box = self.closest_distance(box_lidar, 0.01)
        dist_goal = -tf.math.log(observations[:, self.sensor_offset_table['goal_dist']])
        box_direction = self.average_direction(box_lidar)
        goal_position = dist_goal * observations[:, self.sensor_offset_table['goal_compass']]
        box_position = dist_box * box_direction
        dist_box_goal = tf.linalg.norm(goal_position - box_position, axis=1)
        return dist_box_goal, dist_box

    def closest_distance(self, lidar_measurement, eps=0.0):
        if self.lidar_max_dist is None:
            return -tf.math.log(tf.reduce_max(lidar_measurement, axis=1) + 1e-100) / self.lidar_exp_gain
        else:
            return tf.minimum(
                self.lidar_max_dist - tf.reduce_max(lidar_measurement, axis=1) * self.lidar_max_dist - eps,
                self.lidar_max_dist)

    def average_direction(self, lidar_measurement):
        angles = (tf.range(self.lidar_num_bins) + 0.5) * 2.0 * np.pi / self.lidar_num_bins
        x = tf.math.cos(angles)
        x = tf.broadcast_to(
            x, (tf.shape(lidar_measurement)[0], tf.shape(x)[0])
        )
        y = tf.math.sin(angles)
        y = tf.broadcast_to(
            y, (tf.shape(lidar_measurement)[0], tf.shape(y)[0])
        )
        averaged_x = tf.reduce_sum(tf.linalg.tensordot(x, lidar_measurement + 1e-7, axis=1))
        averaged_y = tf.reduce_sum(tf.linalg.tensordot(y, lidar_measurement + 1e-7, axis=1))
        return tf.stack((averaged_x, averaged_y), axis=1)
