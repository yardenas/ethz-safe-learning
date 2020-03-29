import tensorflow as tf
import numpy as np


class SafetyGymStateScorer(object):
    def __init__(self, config={}):
        for key, value in config.items():
            setattr(self, key, value)
        self.last_dist_goal = 1e3
        self.last_dist_box = 1e3
        self.last_box_goal = 1e3

    def reset(self):
        self.last_dist_goal = 1e3
        self.last_dist_box = 1e3
        self.last_box_goal = 1e3

    def reward(self, observations, actions):
        """ Calculate the dense component of reward.  Call exactly once per step """
        reward = 0.0
        # Distance from robot to goal
        if self.task in ['goal', 'button']:
            dist_goal = self.closest_dist(observations)
            reward += (self.last_dist_goal - dist_goal) * self.reward_distance + \
                      dist_goal <= self.goal_size
            self.last_dist_goal = dist_goal
        # Distance from robot to box
        elif self.task == 'push':
            box_index = np.argmax(
                observations[:, 3 + self.lidar_num_bins:3 + 2 * self.lidar_num_bins]
            )
            dist_box = self.closest_dist(observations[:, box_index])
            goal_index = np.argmax(
                observations[:,
                3 + 2 * self.lidar_num_bins:3 + 3 * self.lidar_num_bins]
            )
            dist_goal = self.closest_dist(observations[:, goal_index])
            gate_dist_box_reward = (self.last_dist_box > self.box_null_dist * self.box_size)
            reward += (self.last_dist_box - dist_box) * self.reward_box_dist * gate_dist_box_reward
            self.last_dist_box = dist_box
        # Distance from box to goal
            box_direction = box_index * 2.0 * np.pi / self.lidar_num_bins
            box_position = np.stack([dist_box * np.cos(box_direction), dist_box * np.sin(box_direction)])
            goal_direction = goal_index * 2.0 * np.pi / self.lidar_num_bins
            goal_position = np.stack([dist_goal * np.cos(goal_direction), dist_goal * np.sin(goal_direction)])
            dist_box_goal = np.norm(goal_position - box_position, axis=1)
            reward += (self.last_box_goal - dist_box_goal) * self.reward_box_goal + \
                dist_box_goal <= self.goal_size
            self.last_box_goal = dist_box_goal
        # Intrinsic reward for uprightness
        if self.reward_orientation:
            accelerometer = observations[:, 2 * self.lidar_num_bins:2 * self.lidar_num_bins + 3] \
                if self.task == 'push' else observations[:, self.lidar_num_bins:self.lidar_num_bins + 3]
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
        cost = 0.0
        # Conctacts processing
        if self.task == 'goal':
            if self.constrain_vases:
                vases_lidar = observations[:, -self.lidar_num_bins:]
                vases_dist = self.closest_dist(vases_lidar)
                cost += vases_dist <= self.vases_size
            if self.constrain_hazards:
                hazards_lidar = observations[:,
                                -2 * self.lidar_num_bins:-self.lidar_num_bins]
                hazards_dist = self.closest_dist(hazards_lidar)
                cost += hazards_dist <= self.hazards_size
        elif self.task == 'push':
            if self.constrain_pillars:
                pillars_lidar = observations[:, -self.lidar_num_bins:]
                pillars_dist = self.closest_dist(pillars_lidar)
                cost += pillars_dist <= self.pillars_size
            if self.constrain_hazards:
                hazards_lidar = observations[:,
                                -2 * self.lidar_num_bins:-self.lidar_num_bins]
                hazards_dist = self.closest_dist(hazards_lidar)
                cost += hazards_dist <= self.hazards_size
        elif self.task == 'button':
            if self.constrain_buttons:
                buttons_lidar = observations[:,
                                3 + self.lidar_num_bins:3 + 2 * self.lidar_num_bins]
                buttons_dist = self.closest_dist(buttons_lidar)
                cost += buttons_dist <= self.buttons_size
            if self.constrain_gremlins:
                gremlins_lidar = observations[:,
                                -2 * self.lidar_num_bins:-self.lidar_num_bins]
                gremlins_dist = self.closest_dist(gremlins_lidar)
                # TODO (yarde): not sure if keepout or size
                cost += gremlins_dist <= self.gremlins_keepout
            if self.constrain_hazards:
                hazards_lidar = observations[:,
                                -2 * self.lidar_num_bins:-self.lidar_num_bins]
                hazards_dist = self.closest_dist(hazards_lidar)
                cost += hazards_dist <= self.hazards_size
        # Displacement processing
        if self.constrain_vases and self.vases_displace_cost:
            print("Should take care of this vases displacement.")

        # Velocity processing
        if self.constrain_vases and self.vases_velocity_cost:
            print("Should take care of this vases velocity.")

        # Optionally remove shaping from reward functions.
        if self.constrain_indicator:
            return cost > 0.0
        return cost

    def closest_dist(self, lidar_measurement):
        if self.lidar_max_dist is None:
            return -np.log(np.max(lidar_measurement, axis=1) + 1e-100) / self.lidar_exp_gain
        else:
            return min(self.lidar_max_dist - np.max(lidar_measurement, axis=1) * self.lidar_max_dist,
                       self.lidar_max_dist)
