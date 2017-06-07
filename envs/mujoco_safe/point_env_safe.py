from rllab.core.serializable import Serializable
#from rllab.envs.mujoco.point_env import PointEnv
from sandbox.cpo.envs.mujoco.point_env import PointEnv
from sandbox.cpo.envs.mujoco_safe.mujoco_env_safe import SafeMujocoEnv
import numpy as np

class SafePointEnv(SafeMujocoEnv, Serializable):

    MODEL_CLASS = PointEnv
