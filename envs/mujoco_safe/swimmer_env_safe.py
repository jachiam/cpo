from rllab.core.serializable import Serializable
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from sandbox.cpo.envs.mujoco_safe.mujoco_env_safe import SafeMujocoEnv
import numpy as np

class SafeSwimmerEnv(SafeMujocoEnv, Serializable):

    MODEL_CLASS = SwimmerEnv

