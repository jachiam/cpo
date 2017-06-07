from rllab.core.serializable import Serializable
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.network import MLP
import lasagne.nonlinearities as NL
import lasagne.layers as L
import numpy as np
import rllab.misc.logger as logger
import theano
import theano.tensor as TT
from rllab.misc.ext import compile_function
from rllab.optimizers.lbfgs_optimizer import LbfgsOptimizer
from rllab.optimizers.first_order_optimizer import FirstOrderOptimizer
from sandbox.cpo.safety_constraints.base import SafetyConstraint


class ParameterizedBonusSafetyConstraint(LasagnePowered, SafetyConstraint, Serializable):

    def __init__(self, wrapped_constraint, 
                       env_spec, 
                       yield_zeros_until=1,
                       optimizer=None, 
                       hidden_sizes=(32,), 
                       hidden_nonlinearity=NL.sigmoid, 
                       lag_time=10, 
                       coeff=1.,
                       filter_bonuses=False,
                       max_epochs=25,
                       *args, **kwargs):

        Serializable.quick_init(self,locals())

        self._wrapped_constraint = wrapped_constraint
        self._env_spec = env_spec
        self._filter_bonuses = filter_bonuses
        self._yield_zeros_until = yield_zeros_until
        self._hidden_sizes = hidden_sizes
        self._lag_time = lag_time
        self._coeff = coeff
        self._max_epochs = max_epochs
        self.use_bonus = True

        if optimizer is None:
            #optimizer = LbfgsOptimizer()
            optimizer = FirstOrderOptimizer(max_epochs=max_epochs, batch_size=None)

        self._optimizer = optimizer

        obs_dim = env_spec.observation_space.flat_dim

        predictor_network = MLP(1,hidden_sizes,hidden_nonlinearity,NL.sigmoid,
                                     input_shape=(obs_dim,))

        LasagnePowered.__init__(self, [predictor_network.output_layer])

        x_var = predictor_network.input_layer.input_var
        y_var = TT.matrix("ys")
        out_var = L.get_output(predictor_network.output_layer, 
                               {predictor_network.input_layer: x_var})

        regression_loss = TT.mean(TT.square(y_var - out_var))

        optimizer_args = dict(
            loss=regression_loss,
            target=self,
            inputs=[x_var, y_var],
        )

        self._optimizer.update_opt(**optimizer_args)
        self._f_predict = compile_function([x_var],out_var)

        self._fit_steps = 0

        self.has_baseline = self._wrapped_constraint.has_baseline
        if self.has_baseline:
            self.baseline = self._wrapped_constraint.baseline

    """
    @property
    def baseline(self):
        return self._wrapped_constraint.baseline
    """

    def evaluate(self, path):
        return self._wrapped_constraint.evaluate(path)

    def get_bonus(self,path):
        if self._fit_steps > self._yield_zeros_until:
            bonus = self._coeff * self._f_predict(path['observations']).reshape(-1)
            if self._filter_bonuses:
                bonus = bonus  * (np.invert(self._wrapped_constraint.evaluate(path)))
            return bonus
        else:
            return np.zeros(path["rewards"].size)

    def prepare_inputs(self, paths):
        xs = []
        ys = []
        for p in paths:
            path_length = len(p["safety_rewards"])
            horizon = min(path_length,self._lag_time)
            xs.append(p["observations"])
            ys.append(np.zeros(path_length))
            for j in range(horizon):
                ys[-1] += np.pad(p["safety_rewards"][j:],[0,j],'constant')
        xs = np.concatenate(xs)
        ys = np.concatenate(ys)
        ys = np.clip(ys,0,1)
        ys = ys.reshape(ys.size,1)
        return [xs,ys]

    def fit(self, paths):
        self._wrapped_constraint.fit(paths)
        prefix = "PredictorBonus"
        inputs = self.prepare_inputs(paths)
        loss_before = self._optimizer.loss(inputs)
        logger.record_tabular(prefix + 'LossBefore', loss_before)
        self._optimizer.optimize(inputs)
        loss_after = self._optimizer.loss(inputs)
        logger.record_tabular(prefix + 'LossAfter', loss_after)
        logger.record_tabular(prefix + 'dLoss', loss_before - loss_after)
        self._fit_steps += 1

    def get_safety_step(self):
        return self._wrapped_constraint.max_value

    def get_param_values(self, **tags):
        return LasagnePowered.get_param_values(self, **tags)

    def set_param_values(self, flattened_params, **tags):
        return LasagnePowered.set_param_values(self, flattened_params, **tags)

