from rllab.core.serializable import Serializable
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from sandbox.cpo.envs.mujoco_safe.mujoco_env_safe import SafeMujocoEnv
import numpy as np

class SafeHalfCheetahEnv(SafeMujocoEnv, Serializable):

    MODEL_CLASS = HalfCheetahEnv

    """
    def get_current_obs(self):
        return np.concatenate([
            self.wrapped_env.model.data.qpos.flatten(), #[1:],
            self.wrapped_env.model.data.qvel.flat,
            self.wrapped_env.get_body_com("torso").flat,
        ])
    """

"""
import os.path as osp
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.misc import logger
from rllab.misc.overrides import overrides


def smooth_abs(x, param):
    return np.sqrt(np.square(x) + np.square(param)) - param


class SafeHalfCheetahEnv(MujocoEnv, Serializable):

    FILE = 'half_cheetah.xml'

    def __init__(self, nonlinear_reward=False, 
                       max_path_length_range=None, 
                       show_limit=True,
                       lim_size=2,
                       xlim=10,
                       *args,
                       **kwargs):
        self._nonlinear_reward = nonlinear_reward
        self._max_path_length_range = max_path_length_range
        self._step = 0

        xml_path = osp.join(MODEL_DIR, self.__class__.FILE)
        if show_limit:
            tree = ET.parse(xml_path)
            worldbody = tree.find(".//worldbody")
            ET.SubElement(
                        worldbody, "geom",
                        name="finish_line",
                        pos="%f %f %f" % (xlim,
                                          0,
                                          lim_size/2),
                        size="%f %f %f" % (0.1,
                                           lim_size,
                                           lim_size),
                        type="box",
                        material="",
                        contype="0",
                        conaffinity="0",
                        rgba="0.1 0.1 0.8 0.4"
            )
            _, file_path = tempfile.mkstemp(text=True)
            tree.write(file_path)
        else:
            file_path = xml_path                

        super(SafeHalfCheetahEnv, self).__init__(*args, file_path=file_path, **kwargs)
        #Serializable.__init__(self, *args, **kwargs)
        Serializable.quick_init(self, locals())

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flatten()[1:],
            self.model.data.qvel.flat,
            self.get_body_com("torso").flat,
        ])

    @overrides
    def reset(self, init_state=None):
        self.reset_mujoco(init_state)
        self.model.forward()
        self.current_com = self.model.data.com_subtree[0]
        self.dcom = np.zeros_like(self.current_com)
        self._step = 0
        if self._max_path_length_range is not None:
            self._last_step = np.random.randint(*self._max_path_length_range)
        return self.get_current_obs()

    def get_body_xmat(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.model.data.xmat[idx].reshape((3, 3))

    def get_body_com(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.model.data.com_subtree[idx]

    def step(self, action):
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()
        action = np.clip(action, *self.action_bounds)
        ctrl_cost = 1e-1 * 0.5 * np.sum(np.square(action))
        run_cost = -1 * self.get_body_comvel("torso")[0]
        cost = ctrl_cost + run_cost
        reward = -cost
        if self._nonlinear_reward:
            reward *= np.abs(reward)
        done = False
        self._step += 1
        if self._max_path_length_range is not None:
            if self._step > self._last_step:
                done = True
        return Step(next_obs, reward, done)

    @overrides
    def log_diagnostics(self, paths):
        progs = [
            path["observations"][-1][-3] - path["observations"][0][-3]
            for path in paths
        ]
        logger.record_tabular('AverageForwardProgress', np.mean(progs))
        logger.record_tabular('MaxForwardProgress', np.max(progs))
        logger.record_tabular('MinForwardProgress', np.min(progs))
        logger.record_tabular('StdForwardProgress', np.std(progs))
"""
