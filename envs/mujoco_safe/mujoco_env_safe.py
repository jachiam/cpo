import os.path as osp
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from rllab import spaces
from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.envs.mujoco.mujoco_env import MujocoEnv, MODEL_DIR
from rllab.envs.proxy_env import ProxyEnv
from rllab.misc import logger
from rllab.misc.overrides import overrides


BIG = 1e6

def smooth_abs(x, param):
    return np.sqrt(np.square(x) + np.square(param)) - param


class SafeMujocoEnv(ProxyEnv, Serializable):

    MODEL_CLASS = None

    def __init__(self, nonlinear_reward=False, 
                       max_path_length_range=None, 
                       random_start_range=None,
                       show_limit=True,
                       lim_size=20,
                       lim_height=2,
                       xlim=10,
                       abs_lim=False,
                       circle_mode=False,
                       target_dist=1.,
                       *args,
                       **kwargs):
        self._nonlinear_reward = nonlinear_reward
        self._max_path_length_range = max_path_length_range
        self._random_start_range = random_start_range
        self._step = 0
        self._circle_mode = circle_mode
        self._target_dist = target_dist


        model_cls = self.__class__.MODEL_CLASS
        if model_cls is None:
            raise "MODEL_CLASS unspecified!"
        xml_path = osp.join(MODEL_DIR, model_cls.FILE)
        if show_limit:
            tree = ET.parse(xml_path)
            worldbody = tree.find(".//worldbody")
            ET.SubElement(
                        worldbody, "geom",
                        name="finish_line",
                        pos="%f %f %f" % (xlim,
                                          0,
                                          lim_height/2),
                        size="%f %f %f" % (0.1,
                                           lim_size,
                                           lim_height),
                        type="box",
                        material="",
                        contype="0",
                        conaffinity="0",
                        rgba="0.1 0.1 0.8 0.4"
            )
            if abs_lim:
                ET.SubElement(
                            worldbody, "geom",
                            name="finish_line2",
                            pos="%f %f %f" % (-xlim,
                                              0,
                                              lim_height/2),
                            size="%f %f %f" % (0.1,
                                               lim_size,
                                               lim_height),
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

        inner_env = model_cls(*args, file_path=file_path, **kwargs)
        ProxyEnv.__init__(self, inner_env)
        Serializable.quick_init(self, locals())
        
        # new code for caching obs / act space
        shp = self.get_current_obs().shape
        ub = BIG * np.ones(shp)
        self._cached_observation_space = spaces.Box(ub * -1, ub)
        bounds = self.wrapped_env.model.actuator_ctrlrange
        lb = bounds[:, 0]
        ub = bounds[:, 1]
        self._cached_action_space = spaces.Box(lb, ub)

    @property
    def observation_space(self):
        #shp = self.get_current_obs().shape
        #ub = BIG * np.ones(shp)
        #return spaces.Box(ub * -1, ub)
        return self._cached_observation_space

    @property
    def action_space(self):
        return self._cached_action_space

    @property
    def action_bounds(self):
        return self.action_space.bounds


    def get_current_obs(self):
        return np.concatenate([
            self.wrapped_env.model.data.qpos.flatten(),
            self.wrapped_env.model.data.qvel.flat,
            self.wrapped_env.get_body_com("torso").flat,
        ])

    def reset(self, init_state=None):
        self.wrapped_env.reset(init_state)
        self._step = 0
        if self._random_start_range is not None:
            l,u = self._random_start_range
            random_start = np.random.rand()*(u - l) + l
            x = self.wrapped_env.model.data.qpos.copy()
            x[0][0] = random_start
            self.wrapped_env.model.data.qpos = x
            self.wrapped_env.model.forward()
        if self._max_path_length_range is not None:
            self._last_step = np.random.randint(*self._max_path_length_range)
        return self.get_current_obs()


    def step(self, action):
        _, reward, done, info = self.wrapped_env.step(action)
        next_obs = self.get_current_obs()

        if self._circle_mode:
            pos = self.wrapped_env.get_body_com("torso")
            vel = self.wrapped_env.get_body_comvel("torso")
            dt = self.wrapped_env.model.opt.timestep
            x, y = pos[0], pos[1]
            dx, dy = vel[0], vel[1]
            reward = -y * dx + x * dy
            reward /= (1 + np.abs( np.sqrt(x **2 + y **2) - self._target_dist))

        if self._nonlinear_reward:
            reward *= np.abs(reward)
        self._step += 1
        if self._max_path_length_range is not None:
            if self._step > self._last_step:
                done = True
        return Step(next_obs, reward, done, **info)

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

        xs = [max(path["observations"][:,-3]) for path in paths]
        logger.record_tabular('AverageFarthestXCoord', np.mean(xs))
        logger.record_tabular('MaxFarthestXCoord', np.max(xs))
        logger.record_tabular('MinFarthestXCoord', np.min(xs))
        logger.record_tabular('StdFarthestXCoord', np.std(xs))

