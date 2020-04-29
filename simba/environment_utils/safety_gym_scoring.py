import numpy as np


class SafetyGymStateScorer(object):
    def __init__(self, config={}):
        for key, value in config.items():
            setattr(self, key, value)
        self.last_dist_goal = 1e3
        self.last_dist_box = 1e3
        self.last_box_goal = 1e3

    def reset(self, observation):
        if self.task in ['goal', 'button']:
            self.last_dist_goal = self.goal_button_distance_metric(observation)
        elif self.task == 'push':
            self.last_box_goal, self.last_dist_box = self.push_distance_metric(observation)

    def reward(self, observations, actions):
        """ Calculate the dense component of reward.  Call exactly once per step """
        if observations.ndim == 1:
            observations = np.expand_dims(observations, axis=0)
            actions = np.expand_dims(actions, axis=0)
        reward = 0.0
        # Distance from robot to goal
        if self.task in ['goal', 'button']:
            dist_goal = self.goal_button_distance_metric(observations)
            reward += (self.last_dist_goal - dist_goal) * self.reward_distance + \
                      (dist_goal <= self.goal_size) * self.reward_goal
            self.last_dist_goal = dist_goal
        # Distance from robot to box
        elif self.task == 'push':
            dist_box_goal, dist_box = self.push_distance_metric(observations)
            reward += (self.last_box_goal - dist_box_goal) * self.reward_box_goal + \
                      (dist_box_goal <= self.goal_size) * self.reward_goal
            self.last_box_goal = dist_box_goal
            gate_dist_box_reward = (self.last_dist_box > self.box_null_dist * self.box_size)
            reward += (self.last_dist_box - dist_box) * self.reward_box_dist * gate_dist_box_reward
            self.last_dist_box = dist_box
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
                vases_dist = self.closest_distance(vases_lidar)
                cost += vases_dist <= self.vases_size
            if self.constrain_hazards:
                hazards_lidar = observations[:,
                                -2 * self.lidar_num_bins:-self.lidar_num_bins]
                hazards_dist = self.closest_distance(hazards_lidar)
                cost += hazards_dist <= self.hazards_size
        elif self.task == 'push':
            if self.constrain_pillars:
                pillars_lidar = observations[:, -self.lidar_num_bins:]
                pillars_dist = self.closest_distance(pillars_lidar)
                cost += pillars_dist <= self.pillars_size
            if self.constrain_hazards:
                hazards_lidar = observations[:,
                                -2 * self.lidar_num_bins:-self.lidar_num_bins]
                hazards_dist = self.closest_distance(hazards_lidar)
                cost += hazards_dist <= self.hazards_size
        elif self.task == 'button':
            if self.constrain_buttons:
                buttons_lidar = observations[:,
                                3 + self.lidar_num_bins:3 + 2 * self.lidar_num_bins]
                buttons_dist = self.closest_distance(buttons_lidar)
                cost += buttons_dist <= self.buttons_size
            if self.constrain_gremlins:
                gremlins_lidar = observations[:,
                                 -2 * self.lidar_num_bins:-self.lidar_num_bins]
                gremlins_dist = self.closest_distance(gremlins_lidar)
                # TODO (yarden): not sure if keepout or size
                cost += gremlins_dist <= self.gremlins_keepout
            if self.constrain_hazards:
                hazards_lidar = observations[:,
                                -2 * self.lidar_num_bins:-self.lidar_num_bins]
                hazards_dist = self.closest_distance(hazards_lidar)
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

    def goal_button_distance_metric(self, observations):
        observations_exp = np.expand_dims(observations, axis=0) if observations.ndim == 1 else \
            observations
        return self.closest_distance(observations_exp[:, 3:3 + self.lidar_num_bins])

    def push_distance_metric(self, observations):
        observations_exp = np.expand_dims(observations, axis=0) if observations.ndim == 1 else \
            observations
        box_observed = np.any(observations_exp[:, 3:3 + self.lidar_num_bins] > 0.0)
        goal_observed = np.any(observations_exp[:,
                               3 + self.lidar_num_bins:3 + 2 * self.lidar_num_bins] > 0.0)
        if goal_observed and box_observed:
            dist_box = self.closest_distance(
                observations_exp[:, 3:3 + self.lidar_num_bins]
            )
            dist_goal = self.closest_distance(
                observations_exp[:, 3 + self.lidar_num_bins:3 + 2 * self.lidar_num_bins]
            )
            # Distance from box to goal
            goal_lidar = observations_exp[:, 3 + self.lidar_num_bins:3 + 2 * self.lidar_num_bins]
            box_lidar = observations_exp[:, 3:3 + self.lidar_num_bins]
            lidar_measurement = np.concatenate((goal_lidar, box_lidar), axis=0)
            directions = self.average_direction(lidar_measurement)
            distances = self.closest_dist(lidar_measurement)
            batch_size = observations_exp.shape[0]
            goal_position = distances[:batch_size] * directions[:batch_size, :]
            box_position = distances[batch_size:2 * batch_size] * \
                           directions[batch_size:2 * batch_size, :]
            dist_box_goal = np.linalg.norm(goal_position - box_position, axis=1)
            return dist_box_goal, dist_box
        elif box_observed:
            fake_max_distance = self.closest_distance(np.zeros((1, 1)))
            return fake_max_distance, self.closest_distance(observations_exp[:, 3:3 + self.lidar_num_bins])
        fake_max_distance = self.closest_distance(np.zeros((1, 1)))
        return fake_max_distance, fake_max_distance

    def closest_distance(self, lidar_measurement):
        if self.lidar_max_dist is None:
            return -np.log(np.max(lidar_measurement, axis=1) + 1e-100) / self.lidar_exp_gain
        else:
            return np.minimum(self.lidar_max_dist - np.max(lidar_measurement, axis=1) * self.lidar_max_dist - 0.018,
                              self.lidar_max_dist)

    def average_direction(self, lidar_measurement):
        angles = np.arange(self.lidar_num_bins) * 2.0 * np.pi / self.lidar_num_bins
        x = np.cos(angles)
        x = np.broadcast_to(
            x, (lidar_measurement.shape[0], x.shape[0])
        )
        y = np.sin(angles)
        y = np.broadcast_to(
            y, (lidar_measurement.shape[0], y.shape[0])
        )
        averaged_x = np.average(x, weights=lidar_measurement, axis=1)
        averaged_y = np.average(y, weights=lidar_measurement, axis=1)
        return np.stack((averaged_x, averaged_y), axis=1)
