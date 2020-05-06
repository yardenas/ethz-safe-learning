import tensorflow as tf
import numpy as np


class SafetyGymStateScorer(object):
    def __init__(self, config={}, obs_space_dict={}):
        for key, value in config.items():
            setattr(self, key, value)
        self.last_dist_goal = 1e3
        self.last_dist_box = 1e3
        self.last_box_goal = 1e3
        self.last_box_observed = False
        self.sensor_offset_table = dict()
        offset = 0
        for k, value in sorted(obs_space_dict.items()):
            k_size = np.prod(value.shape)
            self.sensor_offset_table[k] = slice(offset, offset + k_size)
            offset += k_size

    def reset(self, observation):
        if self.task == 'goal':
            self.last_dist_goal = self.goal_distance_metric(observation)
        elif self.task == 'push':
            self.last_box_goal, self.last_dist_box = self.push_distance_metric(observation)

    def reward(self, observations, actions):
        """ Calculate the dense component of reward.  Call exactly once per step """
        observations_exp = np.expand_dims(observations, axis=0) if observations.ndim == 1 else \
            observations
        actions_exp = np.expand_dims(actions, axis=0) if actions.ndim == 1 else \
            actions
        reward = 0.0
        # Distance from robot to goal
        if self.task == 'goal':
            dist_goal = self.goal_distance_metric(observations_exp)
            reward += (self.last_dist_goal - dist_goal) * self.reward_distance + \
                (dist_goal <= self.goal_size) * self.reward_goal
            self.last_dist_goal = dist_goal
        # Distance from robot to box
        elif self.task == 'push':
            box_observed = np.any(
                observations_exp[:, self.sensor_offset_table['box_lidar']] > 0.0, axis=1)
            rewards_gate = np.logical_and(box_observed, self.last_box_observed)
            dist_box_goal, dist_box = self.push_distance_metric(observations_exp)
            reward += ((self.last_box_goal - dist_box_goal) * self.reward_box_goal + \
                      (dist_box_goal <= self.goal_size) * self.reward_goal) * rewards_gate
            self.last_box_goal = dist_box_goal
            gate_dist_box_reward = (self.last_dist_box > self.box_null_dist * self.box_size)
            reward += ((self.last_dist_box - dist_box) * self.reward_box_dist * gate_dist_box_reward) * rewards_gate
            self.last_dist_box = dist_box
            self.last_box_observed = box_observed
        # Intrinsic reward for uprightness
        if self.reward_orientation:
            accelerometer = observations_exp[:, self.sensor_offset_table['acceleration']]
            zalign = (accelerometer / np.linalg.norm(accelerometer, axis=1, keepdims=True))
            reward += self.reward_orientation_scale * zalign.dot([0.0, 0.0, 1.0])
        # Clip reward
        if self.reward_clip:
            in_range = self.reward_clip > reward > -self.reward_clip
            if not in_range:
                reward = np.clip(reward, -self.reward_clip, self.reward_clip)
                print('Warning: reward was outside of range!')
        return reward

    def cost(self, observations, actions):
        """ Calculate the current costs and return a dict
         assumes SG6 tasks"""
        observations_exp = np.expand_dims(observations, axis=0) if observations.ndim == 1 else \
            observations
        cost = 0.0
        # Conctacts processing
        if self.constrain_vases:
            vases_lidar = observations_exp[:, self.sensor_offset_table['vases_lidar']]
            vases_dist = self.closest_distance(vases_lidar)
            cost += (vases_dist <= self.vases_size)
        if self.constrain_hazards:
            hazards_lidar = observations_exp[:, self.sensor_offset_table['hazards_lidar']]
            hazards_dist = self.closest_distance(hazards_lidar)
            cost += hazards_dist <= self.hazards_size
            # print("fake hazards ", hazards_dist)
        if self.constrain_pillars:
            pillars_lidar = observations_exp[:, self.sensor_offset_table['pillars_lidar']]
            pillars_dist = self.closest_distance(pillars_lidar)
            cost += (pillars_dist <= self.pillars_size)
        if self.constrain_gremlins:
            gremlins_lidar = observations_exp[:, self.sensor_offset_table['gremlins_lidar']]
            gremlins_dist = self.closest_distance(gremlins_lidar)
            cost += (hazards_dist <= self.gremlins_lidar)
        # Displacement processing
        if self.constrain_vases and self.vases_displace_cost:
            print("Should take care of this vases displacement.")
        # Velocity processing
        if self.constrain_vases and self.vases_velocity_cost:
            print("Should take care of this vases velocity.")
        # Optionally remove shaping from reward functions.
        if self.constrain_indicator:
            return int(cost > 0.0)
        return cost

    def goal_distance_metric(self, observations):
        observations_exp = np.expand_dims(observations, axis=0) if observations.ndim == 1 else \
            observations
        return -np.log(observations_exp[:, self.sensor_offset_table['goal_dist']])

    def push_distance_metric(self, observations):
        observations_exp = np.expand_dims(observations, axis=0) if observations.ndim == 1 else \
            observations
        box_lidar = observations_exp[:, self.sensor_offset_table['box_lidar']]
        dist_box = self.closest_distance(box_lidar, 0.01)
        dist_goal = -np.log(observations_exp[:, self.sensor_offset_table['goal_dist']])
        box_direction = self.average_direction(box_lidar)
        box_true_direction = observations_exp[:, self.sensor_offset_table['box_compass']]
        goal_position = dist_goal * observations_exp[:, self.sensor_offset_table['goal_compass']]
        box_position = dist_box * box_direction
        dist_box_goal = np.linalg.norm(goal_position - box_position, axis=1)
        return dist_box_goal, dist_box

    def closest_distance(self, lidar_measurement, eps=0.0):
        if self.lidar_max_dist is None:
            return -np.log(np.max(lidar_measurement, axis=1) + 1e-100) / self.lidar_exp_gain
        else:
            return np.minimum(self.lidar_max_dist - np.max(lidar_measurement, axis=1) * self.lidar_max_dist - eps,
                       self.lidar_max_dist)

    def average_direction(self, lidar_measurement):
        angles = (np.arange(self.lidar_num_bins) + 0.5) * 2.0 * np.pi / self.lidar_num_bins
        x = np.cos(angles)
        x = np.broadcast_to(
            x, (lidar_measurement.shape[0], x.shape[0])
        )
        y = np.sin(angles)
        y = np.broadcast_to(
            y, (lidar_measurement.shape[0], y.shape[0])
        )
        averaged_x = np.average(x, weights=lidar_measurement + 1e-7, axis=1)
        averaged_y = np.average(y, weights=lidar_measurement + 1e-7, axis=1)
        return np.stack((averaged_x, averaged_y), axis=1)
