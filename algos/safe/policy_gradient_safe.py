import gc
import numpy as np
import time
from rllab.algos.batch_polopt import BatchPolopt
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from rllab.core.serializable import Serializable
from rllab.sampler import parallel_sampler
from rllab.misc.overrides import overrides
from rllab.misc import ext
from rllab.misc import special
from rllab.misc import tensor_utils
from rllab.algos import util
import rllab.misc.logger as logger
import theano
import theano.tensor as TT


from sandbox.cpo.algos.safe.sampler_safe import BatchSamplerSafe

class PolicyGradientSafe(BatchPolopt, Serializable):
    """
    Policy Gradient base algorithm

    with optional data reuse and importance sampling,
    and exploration bonuses

    also with safety constraints

    Can use this as a base class for VPG, ERWR, TNPG, TRPO, etc. by picking appropriate optimizers / arguments

    for VPG: use FirstOrderOptimizer
    for ERWR: set positive_adv to True
    for TNPG: use ConjugateGradient optimizer with max_backtracks=1
    for TRPO: use ConjugateGradient optimizer with max_backtracks>1
    for PPO: use PenaltyLBFGS optimzer

    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            all_paths=True,
            step_size=0.01,
            safety_constrained_optimizer=True,
            safety_constraint=None,
            safety_key='advantages',
            safety_discount=1,
            safety_gae_lambda=1,
            center_safety_vals=True,
            robustness_coeff=0.,
            attempt_feasible_recovery=True,
            attempt_infeasible_recovery=True,
            revert_to_last_safe_point=False,
            safety_tradeoff=False,
            safety_tradeoff_coeff=0,
            learn_safety_tradeoff_coeff=False,
            safety_tradeoff_coeff_lr=1e-2,
            pdo_vf_mode=1,
            entropy_regularize=False,
            entropy_coeff=1e-4,
            entropy_coeff_decay=1,
            exploration_bonus=None,
            exploration_lambda=0.001,
            normalize_bonus=True,
            nonnegative_bonus_mean=False,
            batch_aggregate_n=1,
            batch_aggregate_coeff=0.5,
            relative_weights=False,
            importance_sampling=False,
            decision_weight_mode='pd',
            clip_IS_coeff_above=False,
            clip_IS_coeff_below=False,
            IS_coeff_upper_bound=5,
            IS_coeff_lower_bound=0,
            **kwargs):


        """
        :param batch_aggregate_n: use this many epochs of data (including current)
        :param batch_aggregate_coeff: used to make contribution of old data smaller. formula:

            If a batch has age j, it is weighted proportionally to

                                          batch_aggregate_coeff ** j,

            with these batch weights normalized.

            If you want every batch to have equal weight, set batch_aggregate_coeff = 1. 

        :param relative_weights: used to make contribution of old data invariant to how many
                                 more or fewer trajectories the old batch may have.
        :param importance_sampling: do or do not use importance sampling to reweight old data
        :param clip_IS_coeff: if true, clip the IS coefficients.
        :param IS_coeff_bound: if clip_IS_coeff, then IS coefficients are bounded by this value. 
        :param decision_weight_mode: either 'pd', per decision, or 'pt', per trajectory

        """

        Serializable.quick_init(self, locals())

        self.optimizer = optimizer
        self.all_paths = all_paths

        # npo
        self.step_size = step_size

        # safety
        self.safety_constrained_optimizer = safety_constrained_optimizer
        self.safety_constraint = safety_constraint
        self.safety_step_size = self.safety_constraint.get_safety_step()
        assert(safety_key in ['rewards','returns','advantages'])
        if safety_key == 'advantages' and not(hasattr(self.safety_constraint,'baseline')):
            logger.log("Warning: selected advantages as safety key without providing baseline.")
            logger.log("Falling back on returns as safety key.")
            safety_key = 'returns'
        self.safety_key = 'safety_'+safety_key
        self.safety_discount = safety_discount
        self.safety_gae_lambda = safety_gae_lambda
        self.center_safety_vals = center_safety_vals
        self.robustness_coeff = robustness_coeff
        self.attempt_feasible_recovery=attempt_feasible_recovery
        self.attempt_infeasible_recovery=attempt_infeasible_recovery
        self.revert_to_last_safe_point=revert_to_last_safe_point

        # safety tradeoff
        self.safety_tradeoff = safety_tradeoff
        self.safety_tradeoff_coeff = 1. * safety_tradeoff_coeff 
        self.learn_safety_tradeoff_coeff = learn_safety_tradeoff_coeff
        self.safety_tradeoff_coeff_lr = safety_tradeoff_coeff_lr
        self.pdo_vf_mode = pdo_vf_mode      #1 = one VF for R + alpha*S 
                                            #2 = two VFs (one for R, one for S)
                                            #Experiments in the paper use mode 1,
                                            #although I tried out both. 
                                            #(Mode 2 seemed less stable.)

        # entropy regularization
        self.entropy_regularize = entropy_regularize
        self.entropy_coeff = entropy_coeff
        self.entropy_coeff_decay = entropy_coeff_decay

        # intrinsic motivation
        self.exploration_bonus = exploration_bonus
        self.exploration_lambda = exploration_lambda
        self.normalize_bonus = normalize_bonus
        self.nonnegative_bonus_mean = nonnegative_bonus_mean

        # importance sampling
        self.importance_sampling = importance_sampling
        self.decision_weight_mode = decision_weight_mode
        self.clip_IS_coeff_above = clip_IS_coeff_above
        self.clip_IS_coeff_below = clip_IS_coeff_below
        self.IS_coeff_upper_bound = IS_coeff_upper_bound
        self.IS_coeff_lower_bound = IS_coeff_lower_bound
        self.batch_aggregate_n = batch_aggregate_n
        self.batch_aggregate_coeff = batch_aggregate_coeff
        self.relative_weights = relative_weights

        super(PolicyGradientSafe, self).__init__(optimizer=optimizer, 
                                                 sampler_cls=BatchSamplerSafe,
                                                 **kwargs)

        # safety tradeoff
        if self.safety_constraint and self.safety_tradeoff and self.pdo_vf_mode == 1:
            self.baseline._target_key = 'tradeoff_returns'
        

    @overrides
    def init_opt(self):
        self.start_time = time.time()
        is_recurrent = int(self.policy.recurrent)
        obs_var = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1 + is_recurrent,
        )
        action_var = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1 + is_recurrent,
        )
        advantage_var = ext.new_tensor(
            'advantage',
            ndim=1 + is_recurrent,
            dtype=theano.config.floatX
        )
        if self.safety_constraint:
            safety_var = ext.new_tensor(
                'safety_vals',
                ndim=1 + is_recurrent,
                dtype=theano.config.floatX
            )
            

        weights_var = ext.new_tensor(
            'weights',
            ndim=1 + is_recurrent,
            dtype=theano.config.floatX
        )
        dist = self.policy.distribution
        old_dist_info_vars = {
            k: ext.new_tensor(
                'old_%s' % k,
                ndim=2 + is_recurrent,
                dtype=theano.config.floatX
            ) for k in dist.dist_info_keys
            }
        old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]

        state_info_vars = {
            k: ext.new_tensor(
                k,
                ndim=2 + is_recurrent,
                dtype=theano.config.floatX
            ) for k in self.policy.state_info_keys
        }
        state_info_vars_list = [state_info_vars[k] for k in self.policy.state_info_keys]

        if is_recurrent:
            valid_var = TT.matrix('valid')
        else:
            valid_var = None

        dist_info_vars = self.policy.dist_info_sym(obs_var, state_info_vars)

        self.dist_info_vars_func = ext.compile_function(
                inputs=[obs_var] + state_info_vars_list,
                outputs=dist_info_vars,
                log_name="dist_info_vars"
            )

        # when we want to get D_KL( pi' || pi) for data that was sampled on 
        # some behavior policy pi_b, where pi' is the optimization variable
        # and pi is the policy of the previous iteration,
        # the dist_info in memory will correspond to pi_b and not pi. 
        # so we have to compute the dist_info for that data on pi, on the fly.

        ent = dist.entropy_sym(dist_info_vars)
        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
        lr = dist.likelihood_ratio_sym(action_var, old_dist_info_vars, dist_info_vars)
        if is_recurrent:
            mean_ent = TT.sum(weights_var * ent * valid_var) / TT.sum(valid_var)
            max_kl = TT.max(kl * valid_var)
            mean_kl = TT.sum(weights_var * kl * valid_var) / TT.sum(valid_var)
            surr_loss = - TT.sum(lr * weights_var * advantage_var * valid_var) / TT.sum(valid_var)
            if self.safety_constraint:
                f_safety = TT.sum(lr * weights_var * safety_var * valid_var) / TT.sum(valid_var)
        else:
            mean_ent = TT.mean(weights_var * ent)
            max_kl = TT.max(kl)
            mean_kl = TT.mean(weights_var * kl)
            surr_loss = - TT.mean(lr * weights_var * advantage_var)
            if self.safety_constraint:
                f_safety = TT.mean(lr * weights_var * safety_var)

        if self.entropy_regularize:
            self.entropy_beta = theano.shared(self.entropy_coeff)
            surr_loss -= self.entropy_beta * mean_ent

        if self.safety_constraint:
            self.safety_gradient_rescale = theano.shared(1.)
            f_safety = self.safety_gradient_rescale * f_safety


        input_list = [
                         obs_var,
                         action_var,
                         advantage_var,
                         weights_var,
                     ]

        if self.safety_constraint:
            input_list.append(safety_var)

        input_list = input_list + state_info_vars_list + old_dist_info_vars_list
        if is_recurrent:
            input_list.append(valid_var)


        if not(self.safety_constrained_optimizer):
            self.optimizer.update_opt(
                loss=surr_loss,
                target=self.policy,
                leq_constraint=(mean_kl, self.step_size),
                inputs=input_list,
                constraint_name="mean_kl"
            )
        else:
            self.optimizer.update_opt(
                loss=surr_loss,
                target=self.policy,
                quad_leq_constraint=(mean_kl, self.step_size),
                lin_leq_constraint=(f_safety, self.safety_step_size),
                inputs=input_list,
                constraint_name_1="mean_kl",
                constraint_name_2="safety",
                using_surrogate=False,
                precompute=True,
                attempt_feasible_recovery=self.attempt_feasible_recovery,
                attempt_infeasible_recovery=self.attempt_infeasible_recovery,
                revert_to_last_safe_point=self.revert_to_last_safe_point
            )


        f_kl = ext.compile_function(
            inputs=input_list,
            outputs=[mean_kl, max_kl],
        )
        self.opt_info = dict(
            f_kl=f_kl,
        )



    @overrides
    def optimize_policy(self, itr, samples_data):
        logger.log('optimizing policy...')
        all_input_values = tuple(ext.extract(
            samples_data,
            "observations", "actions", "advantages", "weights"
            ))
        if self.safety_constraint:
            all_input_values += tuple(ext.extract(samples_data,"safety_values"))
            self.safety_gradient_rescale.set_value(samples_data['safety_rescale'])
            logger.record_tabular('SafetyGradientRescale', self.safety_gradient_rescale.get_value())
            """
            I think this one is worth some explanation. The surrogate function is computed by taking
            an average of likelihood ratios times safety advantages. IE, it is a sample expectation 
            over state-action pairs. Suppose we have N trajectories of length T. Then the surrogate is

                surrogate = (1 / NT) * sum_{j=1}^N sum_{t=1}^T lr(j,t) * adv(j,t)

            But the true safety constraint function is an expectation over /trajectories/, not state-action
            pairs. 

                true constraint = (1 / N) * sum_{j=1}^N sum_{t=1}^T lr(j,t) * adv(j,t)
                                = T * surrogate

            So the gradient of the surrogate is (1 / T) times the gradient of the true constraint. 
            In normal policy gradient situations, this isn't a problem, because we only care about the
            direction and not the magnitude. But, our safety constraint formulation crucially relies
            on this gradient having the correct magnitude! So we have to rescale appropriately. 
            The "SafetyGradientRescale" is automatically computed by the sampler and provided to 
            the optimizer.
            """

        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
        all_input_values += tuple(state_info_list) + tuple(dist_info_list)
        if self.policy.recurrent:
            all_input_values += (samples_data["valids"],)
        loss_before = self.optimizer.loss(all_input_values)


        if not(self.safety_constrained_optimizer):
            self.optimizer.optimize(all_input_values)
        else:
            threshold = max(self.safety_step_size - samples_data['safety_eval'],0)
            if 'advantage' in self.safety_key:
                std_adv = np.std(samples_data["safety_values"])
                logger.record_tabular('StdSafetyAdv',std_adv)
                threshold = max(threshold - self.robustness_coeff*std_adv,0)
            
            if 'safety_offset' in samples_data:
                logger.record_tabular('SafetyOffset',samples_data['safety_offset'])

            self.optimizer.optimize(all_input_values,
                    precomputed_eval = samples_data['safety_eval'],
                    precomputed_threshold = threshold,
                    diff_threshold=True)

        mean_kl, max_kl = self.opt_info['f_kl'](*all_input_values)
        loss_after = self.optimizer.loss(all_input_values)

        if self.entropy_regularize and not(self.entropy_coeff_decay == 1):
            current_entropy_coeff = self.entropy_beta.get_value() * self.entropy_coeff_decay
            self.entropy_beta.set_value(current_entropy_coeff)
            logger.record_tabular('EntropyCoeff', current_entropy_coeff)


        if self.learn_safety_tradeoff_coeff:
            delta = samples_data['safety_eval'] - self.safety_step_size
            logger.record_tabular('TradeoffCoeffBefore',self.safety_tradeoff_coeff)
            self.safety_tradeoff_coeff += self.safety_tradeoff_coeff_lr * delta
            self.safety_tradeoff_coeff = max(0, self.safety_tradeoff_coeff)
            logger.record_tabular('TradeoffCoeffAfter',self.safety_tradeoff_coeff)
            
        logger.record_tabular('Time',time.time() - self.start_time)
        logger.record_tabular('LossBefore', loss_before)
        logger.record_tabular('LossAfter', loss_after)
        logger.record_tabular('MeanKL', mean_kl)
        logger.record_tabular('MaxKL', max_kl)
        logger.record_tabular('dLoss', loss_before - loss_after)
        logger.log('optimization finished')


    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
            expl=self.exploration_bonus,
            safe=self.safety_constraint
        )
