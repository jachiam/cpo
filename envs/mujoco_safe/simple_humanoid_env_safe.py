from rllab.core.serializable import Serializable
from rllab.envs.mujoco.simple_humanoid_env import SimpleHumanoidEnv
from sandbox.cpo.envs.mujoco_safe.mujoco_env_safe import SafeMujocoEnv
from rllab.envs.base import Step
import numpy as np

class SafeSimpleHumanoidEnv(SafeMujocoEnv, Serializable):

    MODEL_CLASS = SimpleHumanoidEnv

    def get_current_obs(self):
        data = self.wrapped_env.model.data
        return np.concatenate([
            data.qpos.flat,
            data.qvel.flat,
            np.clip(data.cfrc_ext, -1, 1).flat,
            self.wrapped_env.get_body_com("torso").flat,
        ])

    def step(self, action):
        self.wrapped_env.forward_dynamics(action)
        next_obs = self.get_current_obs()


        alive_bonus = self.wrapped_env.alive_bonus
        data = self.wrapped_env.model.data

        comvel = self.wrapped_env.get_body_comvel("torso")

        lin_vel_reward = comvel[0]
        vel_deviation_cost = 0.5 * self.wrapped_env.vel_deviation_cost_coeff * np.sum(
            np.square(comvel[1:]))

        vel_reward = lin_vel_reward - vel_deviation_cost

        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = .5 * self.wrapped_env.ctrl_cost_coeff * np.sum(
            np.square(action / scaling))
        impact_cost = .5 * self.wrapped_env.impact_cost_coeff * np.sum(
            np.square(np.clip(data.cfrc_ext, -1, 1)))


        done = data.qpos[2] < 0.8 or data.qpos[2] > 2.0


        if self._circle_mode:
            pos = self.wrapped_env.get_body_com("torso")
            vel = self.wrapped_env.get_body_comvel("torso")
            dt = self.wrapped_env.model.opt.timestep
            x, y = pos[0], pos[1]
            dx, dy = vel[0], vel[1]
            vel_reward = -y * dx + x * dy
            vel_reward /= (1 + np.abs( np.sqrt(x **2 + y **2) - self._target_dist))

        reward = vel_reward + alive_bonus - ctrl_cost - impact_cost


        self._step += 1
        if self._max_path_length_range is not None:
            if self._step > self._last_step:
                done = True
        return Step(next_obs, reward, done)
