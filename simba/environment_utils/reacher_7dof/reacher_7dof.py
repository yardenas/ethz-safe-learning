import os
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


# Copy-pasted from Berkeley CS285 course and then adapted to this project.
class Reacher7DOFEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.hand_sid = -2
        self.target_sid = -1
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        mujoco_env.MujocoEnv.__init__(self, curr_dir + '/assets/sawyer.xml', 2)
        utils.EzPickle.__init__(self)
        self.observation_dim = 26
        self.action_dim = 7
        self.hand_sid = self.model.site_name2id("finger")
        self.target_sid = self.model.site_name2id("target")
        self.skip = self.frame_skip
        self.reset_pose = self.init_qpos.copy()
        self.reset_vel = self.init_qvel.copy()
        self.reset_goal = np.zeros(3)

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos.flat,
            self.data.qvel.flatten() / 10.,
            self.data.site_xpos[self.hand_sid],
            self.model.site_pos[self.target_sid],
        ])

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        reward, done = self.reward(np.expand_dims(ob, axis=0),
                                   np.expand_dims(a, axis=0))
        return ob, reward, bool(done), {}

    def reward(self, observations, actions):
        hand_pos = observations[:, -6:-3]
        target_pos = observations[:, -3:]
        dist = np.linalg.norm(hand_pos - target_pos, axis=1)
        dones = np.zeros((observations.shape[0],), dtype=np.bool)
        return -10.0 * dist, dones

    def reset(self):
        _ = self.reset_model()
        self.model.site_pos[self.target_sid] = [0.1, 0.1, 0.1]
        _, _reward, _, info = self.step(np.zeros(7))
        ob = self._get_obs()
        return ob

    def reset_model(self):
        self.reset_pose = self.init_qpos.copy()
        self.reset_vel = self.init_qvel.copy()
        self.reset_goal = np.zeros(3)
        self.reset_goal[0] = self.np_random.uniform(low=-0.3, high=0.3)
        self.reset_goal[1] = self.np_random.uniform(low=-0.2, high=0.2)
        self.reset_goal[2] = self.np_random.uniform(low=-0.25, high=0.25)
        return self.do_reset(self.reset_pose, self.reset_vel, self.reset_goal)

    def do_reset(self, reset_pose, reset_vel, reset_goal):
        self.set_state(reset_pose, reset_vel)
        self.reset_goal = reset_goal.copy()
        self.model.site_pos[self.target_sid] = self.reset_goal
        self.sim.forward()
        return self._get_obs()
