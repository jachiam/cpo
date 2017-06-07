from rllab.envs.base import Step
from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides
import numpy as np
import math
from rllab.mujoco_py import glfw


class PointEnv(MujocoEnv, Serializable):

    """
    Use Left, Right, Up, Down, A (steer left), D (steer right)
    """

    FILE = 'point.xml'

    def __init__(self, 
            size=40, 
            align_mode=True,
            reward_dir=[0.,0.], 
            target_dist=5.,
            *args, **kwargs):
        self.size = size
        self.align_mode = align_mode
        self.reward_dir = reward_dir
        self.target_dist = target_dist
        super(PointEnv, self).__init__(*args, **kwargs)
        Serializable.quick_init(self, locals())

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flatten(),
            self.model.data.qvel.flat,
            self.get_body_com("torso").flat,
        ])

    def step(self, action):
        qpos = np.copy(self.model.data.qpos)
        qpos[2, 0] += action[1]
        ori = qpos[2, 0]
        # compute increment in each direction
        dx = math.cos(ori) * action[0]
        dy = math.sin(ori) * action[0]
        # ensure that the robot is within reasonable range
        qpos[0, 0] = np.clip(qpos[0, 0] + dx, -self.size, self.size)
        qpos[1, 0] = np.clip(qpos[1, 0] + dy, -self.size, self.size)
        self.model.data.qpos = qpos
        self.model.forward()
        next_obs = self.get_current_obs()
        if self.align_mode:
            reward = max(self.reward_dir[0] * dx + self.reward_dir[1] * dy,0)
        else:
            x, y = qpos[0,0], qpos[1,0]
            reward = -y * dx + x * dy
            reward /= (1 + np.abs( np.sqrt(x **2 + y **2) - self.target_dist))
        return Step(next_obs, reward, False)

    def get_xy(self):
        qpos = self.model.data.qpos
        return qpos[0, 0], qpos[1, 0]

    def set_xy(self, xy):
        qpos = np.copy(self.model.data.qpos)
        qpos[0, 0] = xy[0]
        qpos[1, 0] = xy[1]
        self.model.data.qpos = qpos
        self.model.forward()

    @overrides
    def action_from_key(self, key):
        lb, ub = self.action_bounds
        if key == glfw.KEY_LEFT:
            return np.array([0, ub[0]*0.3])
        elif key == glfw.KEY_RIGHT:
            return np.array([0, lb[0]*0.3])
        elif key == glfw.KEY_UP:
            return np.array([ub[1], 0])
        elif key == glfw.KEY_DOWN:
            return np.array([lb[1], 0])
        else:
            return np.array([0, 0])

