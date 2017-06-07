import numpy as np
import rllab.misc.logger as logger
from rllab.sampler import parallel_sampler
from rllab.sampler.base import Sampler
from rllab.misc import ext
from rllab.misc import special
from rllab.misc import tensor_utils
from rllab.algos import util



def local_truncate_paths(paths, max_samples):
    """
    Truncate the list of paths so that the total number of samples is almost equal to max_samples. This is done by
    removing extra paths at the end of the list. But here, we do NOT make the last path shorter.
    :param paths: a list of paths
    :param max_samples: the absolute maximum number of samples
    :return: a list of paths, truncated so that the number of samples adds up to max-samples
    """
    # chop samples collected by extra paths
    # make a copy
    paths = list(paths)
    total_n_samples = sum(len(path["rewards"]) for path in paths)
    while len(paths) > 0 and total_n_samples - len(paths[-1]["rewards"]) >= max_samples:
        total_n_samples -= len(paths.pop(-1)["rewards"])
    return paths



class BatchSamplerSafe(Sampler):
    def __init__(self, algo, **kwargs):
        """
        :type algo: BatchPolopt
        """
        self.algo = algo
        self.experience_replay = []     # list of sample batches
        self.env_interacts_memory = []  # list of how many env interacts are in each batch
        self.env_interacts = 0
        self.total_env_interacts = 0
        self.mean_path_len = 0
        self.use_safety_bonus = (self.algo.safety_constraint and
                                 hasattr(self.algo.safety_constraint,'get_bonus') and 
                                 self.algo.safety_constraint.use_bonus)

        self.use_safety_baselines = (self.algo.safety_constraint and 
                                     self.algo.safety_key == 'safety_advantages' and 
                                     hasattr(self.algo.safety_constraint,'baseline'))


    def start_worker(self):
        parallel_sampler.populate_task(self.algo.env, self.algo.policy, scope=self.algo.scope)

    def shutdown_worker(self):
        parallel_sampler.terminate_task(scope=self.algo.scope)

    def obtain_samples(self, itr):
        cur_params = self.algo.policy.get_param_values()
        paths = parallel_sampler.sample_paths(
            policy_params=cur_params,
            max_samples=self.algo.batch_size,
            max_path_length=self.algo.max_path_length,
            scope=self.algo.scope,
        )

        """log_likelihoods for importance sampling"""
        for path in paths:
            logli = self.algo.policy.distribution.log_likelihood(path["actions"],path["agent_infos"])
            path["log_likelihood"] = logli


        """keep data use per iteration approximately fixed"""
        if not(self.algo.all_paths):
            paths = local_truncate_paths(paths, self.algo.batch_size)

        """keep track of path length"""
        self.env_interacts = sum([len(path["rewards"]) for path in paths])
        self.total_env_interacts += self.env_interacts
        self.mean_path_len = float(self.env_interacts)/len(paths)

        """manage experience replay for old batch reuse"""
        self.experience_replay.append(paths)
        self.env_interacts_memory.append(self.env_interacts)
        if len(self.experience_replay) > self.algo.batch_aggregate_n:
            self.experience_replay.pop(0)
            self.env_interacts_memory.pop(0)

        return paths


    def process_samples(self, itr, paths):

        """
        we will ignore paths argument and only use experience replay.
        note: if algo.batch_aggregate_n = 1, then the experience replay will
        only contain the most recent batch, and so len(all_paths) == 1.
        """
        
        if self.algo.exploration_bonus:
            self.compute_exploration_bonuses_and_statistics()

        if self.algo.safety_constraint:
            self.compute_safety_function_and_statistics()

        self.compute_epoch_weights()

        all_paths = []
        all_evs = []

        for paths in self.experience_replay:
            batch_ev = self.process_single_batch(paths)
            all_paths += paths
            all_evs.append(batch_ev)

        all_evs = all_evs[::-1]     # most recent is now at element 0

        """
        importance sampling if old data is used
        """
        if self.algo.batch_aggregate_n > 1 and self.algo.importance_sampling:
            self.compute_all_importance_weights(ignore_age_0=True)

        samples_data = self.create_samples_dict(all_paths)

        """log all useful info"""
        self.record_statistics(itr, all_paths, all_evs, samples_data)

        """update vf, exploration bonus, and safety model"""
        self.update_parametrized_models()

        return samples_data


    def compute_exploration_bonuses_and_statistics(self):

        for paths in self.experience_replay: 
            for path in paths:
                path["bonuses"] = self.algo.exploration_bonus.get_bonus(path)

        """ total and mean over all of memory """
        self.bonus_total =  sum([
                                sum([
                                    sum(path["bonuses"])
                                for path in paths])
                            for paths in self.experience_replay])

        self.bonus_mean = self.bonus_total / sum(self.env_interacts_memory)

        """ total and mean over most recent batch of data """
        self.new_bonus_total = sum([sum(path["bonuses"]) for path in self.experience_replay[-1]])
        self.new_bonus_mean = self.new_bonus_total / self.env_interacts_memory[-1]

        self.bonus_baseline = self.algo.exploration_lambda * \
                              min(0,self.bonus_mean / max(1,np.abs(self.bonus_mean)))


    def compute_safety_function_and_statistics(self):

        for paths in self.experience_replay: 
            for path in paths:
                path["safety_rewards"] = self.algo.safety_constraint.evaluate(path)
                if (hasattr(self.algo.safety_constraint,'get_bonus') and
                    self.algo.safety_constraint.use_bonus):
                    path["safety_bonuses"] = self.algo.safety_constraint.get_bonus(path)

    
    def compute_epoch_weights(self):
        """create weights, with highest weight on most recent batch"""
        self.raw_weights = np.array(
                        [self.algo.batch_aggregate_coeff**j for j in range(len(self.experience_replay))],
                        dtype='float'
                        )
        self.raw_weights /= sum(self.raw_weights)
        self.raw_weights = self.raw_weights[::-1]
        self.weights = self.raw_weights.copy()

        if self.algo.relative_weights:
            """reweight the weights by how many paths are in that batch """
            total_paths = sum([len(paths) for paths in self.experience_replay])
            for j in range(len(self.weights)):
                self.weights[j] *= total_paths / len(self.experience_replay[j])

        self.age = np.arange(len(self.experience_replay))[::-1]
        

    def process_single_batch(self, paths):

        if hasattr(self.algo.baseline, "predict_n"):
            all_path_baselines = self.algo.baseline.predict_n(paths)
        else:
            all_path_baselines = [self.algo.baseline.predict(path) for path in paths]

        if self.use_safety_baselines:
            all_path_safety_baselines = \
                [self.algo.safety_constraint.baseline.predict(path) for path in paths]
        

        for idx, path in enumerate(paths):
            if "weights" not in path:
                path["weights"] = np.ones_like(path["rewards"])

            path_baselines = np.append(all_path_baselines[idx], 0)
            deltas = path["rewards"] + \
                     self.algo.discount * path_baselines[1:] - \
                     path_baselines[:-1]

            """exploration bonuses"""
            if self.algo.exploration_bonus:
                path["bonuses"] *= self.algo.exploration_lambda
                if self.algo.normalize_bonus:
                    path["bonuses"] /= max(1,np.abs(self.bonus_mean))
                if self.algo.nonnegative_bonus_mean:
                    path["bonuses"] -= self.bonus_baseline
                deltas += path["bonuses"]

            path["advantages"] = special.discount_cumsum(
                deltas, self.algo.discount * self.algo.gae_lambda)
            path["returns"] = special.discount_cumsum(path["rewards"], self.algo.discount)

            """safety constraint values"""
            if self.algo.safety_constraint:

                path["safety_returns"] = \
                    special.discount_cumsum(path["safety_rewards"],self.algo.safety_discount)
                
                if self.use_safety_bonus:
                    path["safety_robust_rewards"] = path["safety_rewards"] + path["safety_bonuses"]     
                    path["safety_robust_returns"] = \
                        special.discount_cumsum(path["safety_robust_rewards"],self.algo.safety_discount)

                if self.use_safety_baselines:
                    path_safety_baselines = np.append(all_path_safety_baselines[idx],0)

                    safety_deltas = path["safety_rewards"] + \
                                    self.algo.safety_discount * path_safety_baselines[1:] - \
                                    path_safety_baselines[:-1]                   

                    path["safety_advantages"] = special.discount_cumsum(
                        safety_deltas, self.algo.safety_discount * self.algo.safety_gae_lambda)

                if self.use_safety_bonus and self.use_safety_baselines:
                    safety_robust_deltas = path["safety_robust_rewards"] + \
                                    self.algo.safety_discount * path_safety_baselines[1:] - \
                                    path_safety_baselines[:-1]                   

                    path["safety_robust_advantages"] = special.discount_cumsum(
                        safety_robust_deltas, 
                        self.algo.safety_discount * self.algo.safety_gae_lambda)  

                if self.algo.safety_tradeoff:
                    if not(self.use_safety_bonus):
                        safety_reward_key = 'safety_rewards'
                    else:
                        safety_reward_key = 'safety_robust_rewards'

                    tradeoff_rewards = path["rewards"] - self.algo.safety_tradeoff_coeff * path[safety_reward_key]
                    path["tradeoff_rewards"] = tradeoff_rewards
                    path["tradeoff_returns"] = special.discount_cumsum(tradeoff_rewards, self.algo.discount)

                    if self.algo.pdo_vf_mode == 1:
                        tradeoff_deltas  = deltas - self.algo.safety_tradeoff_coeff * path[safety_reward_key]
                        path["advantages"] = special.discount_cumsum(
                            tradeoff_deltas, self.algo.discount * self.algo.gae_lambda)
                    else:
                        if not(self.use_safety_bonus):
                            tradeoff_deltas = deltas - self.algo.safety_tradeoff_coeff * safety_deltas
                        else:
                            tradeoff_deltas = deltas - self.algo.safety_tradeoff_coeff * safety_robust_deltas

                        path["advantages"] = special.discount_cumsum(
                            tradeoff_deltas, self.algo.discount * self.algo.gae_lambda)

        ev = special.explained_variance_1d(
                np.concatenate(all_path_baselines),
                np.concatenate([path[self.algo.baseline._target_key] for path in paths])
                )                      

        return ev

    def compute_all_importance_weights(self,ignore_age_0=False):
        """record the IS_coeffs"""
        self.IS_coeffs = [[] for paths in self.experience_replay]
        for paths, weight, age in zip(self.experience_replay,self.weights,self.age):
            if age==0 and ignore_age_0:
                continue
            for path in paths:
                path["weights"] = weight * np.ones_like(path["rewards"])
                self.update_agent_infos(path)
                self.compute_and_apply_importance_weights(path)
                path["weights"] *= path["IS_coeff"]
            self.IS_coeffs[age] = [path["IS_coeff"] for path in paths]


    # unused, for debug only
    def compute_batch_importance_weights(self,paths,weight=1):
        for path in paths:
            """recompute agent infos for old data"""
            """(necessary for correct reuse of old data)"""
            path["weights"] = weight * np.ones_like(path["rewards"])
            self.update_agent_infos(path)
            self.compute_and_apply_importance_weights(path)
            path["weights"] *= path["IS_coeff"]
 

    def update_agent_infos(self,path):
        """
        this updates the agent dist infos (i.e, mean & variance of Gaussian policy dist)
        so that it can compute the probability of taking these actions on the most recent
        policy is.
        meanwhile, the log likelihood of taking the actions on the original behavior policy
        can still be found in path["log_likelihood"].
        """
        state_info_list = [path["agent_infos"][k] for k in self.algo.policy.state_info_keys]
        input_list = tuple([path["observations"]] + state_info_list)
        cur_dist_info = self.algo.dist_info_vars_func(*input_list)
        for k in self.algo.policy.distribution.dist_info_keys:
            path["agent_infos"][k] = cur_dist_info[k]


    def compute_and_apply_importance_weights(self,path):
        new_logli = self.algo.policy.distribution.log_likelihood(path["actions"],path["agent_infos"])
        logli_diff = new_logli - path["log_likelihood"]
        if self.algo.decision_weight_mode=='pd':
            logli_diff = logli_diff[::-1]
            log_decision_weighted_IS_coeffs = special.discount_cumsum(logli_diff,1)
            IS_coeff = np.exp(log_decision_weighted_IS_coeffs[::-1])
        elif self.algo.decision_weight_mode=='pt':
            IS_coeff = np.exp(np.sum(logli_diff))
        if self.algo.clip_IS_coeff_above:
            IS_coeff = np.minimum(IS_coeff,self.algo.IS_coeff_upper_bound)
        if self.algo.clip_IS_coeff_below:
            IS_coeff = np.maximum(IS_coeff,self.algo.IS_coeff_lower_bound)

        path["IS_coeff"] = IS_coeff



    def create_samples_dict(self, paths):

        if self.algo.safety_constraint:
            if self.use_safety_bonus:
                safety_key = 'safety_robust' + self.algo.safety_key[6:]
            else:
                safety_key = self.algo.safety_key

            logger.log("Policy optimization is using safety_key=%s." % safety_key)

        if not self.algo.policy.recurrent:
            observations = tensor_utils.concat_tensor_list([path["observations"] for path in paths])
            actions = tensor_utils.concat_tensor_list([path["actions"] for path in paths])
            rewards = tensor_utils.concat_tensor_list([path["rewards"] for path in paths])
            returns = tensor_utils.concat_tensor_list([path["returns"] for path in paths])
            advantages = tensor_utils.concat_tensor_list([path["advantages"] for path in paths])
            env_infos = tensor_utils.concat_tensor_dict_list([path["env_infos"] for path in paths])
            agent_infos = tensor_utils.concat_tensor_dict_list([path["agent_infos"] for path in paths])
            weights = tensor_utils.concat_tensor_list([path["weights"] for path in paths])

            if self.algo.center_adv:
                advantages = util.center_advantages(advantages)

            if self.algo.positive_adv:
                advantages = util.shift_advantages_to_positive(advantages)

            samples_data = dict(
                observations=observations,
                actions=actions,
                rewards=rewards,
                returns=returns,
                advantages=advantages,
                env_infos=env_infos,
                agent_infos=agent_infos,
                weights=weights,
                paths=paths,
            )

            if self.algo.safety_constraint:

                safety_vals = tensor_utils.concat_tensor_list([path[safety_key] for path in paths])
                samples_data['safety_values'] = safety_vals     # for gradient calculation
                if self.algo.center_safety_vals:
                    samples_data['safety_offset'] = np.mean(safety_vals)
                    samples_data['safety_values'] = samples_data['safety_values'] - samples_data['safety_offset']

        else:
            max_path_length = max([len(path["advantages"]) for path in paths])

            # make all paths the same length (pad extra advantages with 0)
            obs = [path["observations"] for path in paths]
            obs = tensor_utils.pad_tensor_n(obs, max_path_length)

            if self.algo.center_adv:
                raw_adv = np.concatenate([path["advantages"] for path in paths])
                adv_mean = np.mean(raw_adv)
                adv_std = np.std(raw_adv) + 1e-8
                adv = [(path["advantages"] - adv_mean) / adv_std for path in paths]
            else:
                adv = [path["advantages"] for path in paths]

            adv = np.asarray([tensor_utils.pad_tensor(a, max_path_length) for a in adv])

            actions = [path["actions"] for path in paths]
            actions = tensor_utils.pad_tensor_n(actions, max_path_length)

            rewards = [path["rewards"] for path in paths]
            rewards = tensor_utils.pad_tensor_n(rewards, max_path_length)

            returns = [path["returns"] for path in paths]
            returns = tensor_utils.pad_tensor_n(returns, max_path_length)

            agent_infos = [path["agent_infos"] for path in paths]
            agent_infos = tensor_utils.stack_tensor_dict_list(
                [tensor_utils.pad_tensor_dict(p, max_path_length) for p in agent_infos]
            )

            env_infos = [path["env_infos"] for path in paths]
            env_infos = tensor_utils.stack_tensor_dict_list(
                [tensor_utils.pad_tensor_dict(p, max_path_length) for p in env_infos]
            )

            weights = [path["weights"] for path in paths]
            weights = tensor_utils.pad_tensor_n(weights, max_path_length)

            valids = [np.ones_like(path["returns"]) for path in paths]
            valids = tensor_utils.pad_tensor_n(valids, max_path_length)

            samples_data = dict(
                observations=obs,
                actions=actions,
                advantages=adv,
                rewards=rewards,
                returns=returns,
                valids=valids,
                agent_infos=agent_infos,
                env_infos=env_infos,
                weights=weights,
                paths=paths,
            )


            if self.algo.safety_constraint:

                safety_vals = [path[safety_key] for path in paths]
                if self.algo.center_safety_vals:
                    samples_data['safety_offset'] = np.mean(safety_vals)
                    safety_vals = safety_vals - samples_data['safety_offset']
                safety_vals = tensor_utils.pad_tensor_n(safety_vals, max_path_length)
                samples_data['safety_values'] = safety_vals

        if self.algo.safety_constraint:
            # logic currently only supports linearization constant calculated on most recent batch of data
            # because importance sampling is complicated
            if self.algo.safety_key == 'safety_rewards':
                if self.use_safety_bonus:
                    key = 'safety_robust_rewards'
                else:
                    key = 'safety_rewards'
                safety_eval = np.mean(tensor_utils.concat_tensor_list(
                                    [path[key] for path in self.experience_replay[-1]]
                                ))
            else:
                if self.use_safety_bonus:
                    key = 'safety_robust_returns'
                else:
                    key = 'safety_returns'
                safety_eval = np.mean(
                                    [path[key][0] for path in self.experience_replay[-1]]
                                )
            samples_data['safety_eval'] = safety_eval       # linearization constant
            samples_data['safety_rescale'] = len(samples_data['safety_values']) / sum([len(paths) for paths in self.experience_replay])


        return samples_data


    def record_statistics(self, itr, paths, evs, samples_data): 

        # Compute statistics for new paths
        average_discounted_return = \
            np.mean([path["returns"][0] for path in self.experience_replay[-1]])

        undiscounted_returns = [sum(path["rewards"]) for path in self.experience_replay[-1]]

        agent_infos = tensor_utils.concat_tensor_dict_list( 
                            [path["agent_infos"] for path in self.experience_replay[-1]]
                            )
        ent = np.mean(self.algo.policy.distribution.entropy(agent_infos))


        # log everything
        logger.record_tabular('Iteration', itr)
        logger.record_tabular('AverageDiscountedReturn',
                              average_discounted_return)
        logger.record_tabular('AverageReturn', np.mean(undiscounted_returns))
        if self.algo.safety_constraint and self.algo.safety_tradeoff:
            average_discounted_tradeoff_return = \
                np.mean([path["tradeoff_returns"][0] for path in self.experience_replay[-1]])

            average_undiscounted_tradeoff_return = \
                np.mean([sum(path["tradeoff_rewards"]) for path in self.experience_replay[-1]])

            logger.record_tabular('AverageDiscountedTradeoffReturn',
                                  average_discounted_tradeoff_return)
            logger.record_tabular('AverageTradeoffReturn',
                                  average_undiscounted_tradeoff_return)
        logger.record_tabular('ExplainedVariance', evs[0])
        logger.record_tabular('NumBatches',len(self.experience_replay))
        logger.record_tabular('NumTrajs', len(paths))
        logger.record_tabular('MeanPathLen',self.mean_path_len)
        #logger.record_tabular('MeanWeight',np.mean(samples_data['weights']))
        logger.record_tabular('EnvInteracts',self.env_interacts)
        logger.record_tabular('TotalEnvInteracts',self.total_env_interacts)
        logger.record_tabular('Entropy', ent)
        logger.record_tabular('Perplexity', np.exp(ent))
        logger.record_tabular('StdReturn', np.std(undiscounted_returns))
        logger.record_tabular('MaxReturn', np.max(undiscounted_returns))
        logger.record_tabular('MinReturn', np.min(undiscounted_returns))

        if self.algo.batch_aggregate_n > 1:
            for age in range(self.algo.batch_aggregate_n):
                if age < len(self.experience_replay):
                    raw_weight = self.raw_weights[::-1][age]
                    weight     = self.weights[::-1][age]
                    logger.record_tabular('RawWeight_age_' + str(age),raw_weight)
                    logger.record_tabular('ScaledWeight_age_' + str(age),weight)
                    if age > 0 and self.algo.importance_sampling:
                        IS = self.get_IS(age)
                        logger.record_tabular('MeanISCoeff_age_' + str(age),np.mean(IS))
                        logger.record_tabular('StdISCoeff_age_' + str(age),np.std(IS))
                        logger.record_tabular('MaxISCoeff_age_' + str(age),np.max(IS))
                        logger.record_tabular('MinISCoeff_age_' + str(age),np.min(IS))
                    logger.record_tabular('ExplainedVariance_age_'+str(age),evs[age])
                else:
                    logger.record_tabular('RawWeight_age_' + str(age),0)
                    logger.record_tabular('ScaledWeight_age_' + str(age),0)
                    if age > 0 and self.algo.importance_sampling:
                        logger.record_tabular('MeanISCoeff_age_' + str(age),0)
                        logger.record_tabular('StdISCoeff_age_' + str(age),0)
                        logger.record_tabular('MaxISCoeff_age_' + str(age),0)
                        logger.record_tabular('MinISCoeff_age_' + str(age),0)
                    logger.record_tabular('ExplainedVariance_age_'+str(age),0)
                

        if self.algo.exploration_bonus:
            bonuses = tensor_utils.concat_tensor_list([path["bonuses"] for path in paths])
            logger.record_tabular('MeanRawBonus',self.bonus_mean)
            logger.record_tabular('MeanBonus',np.mean(bonuses))
            logger.record_tabular('StdBonus',np.std(bonuses))
            logger.record_tabular('MaxBonus',np.max(bonuses))
            bonus_sums = np.array([np.sum(path["bonuses"]) for path in paths])
            logger.record_tabular('MeanBonusSum', np.mean(bonus_sums))
            logger.record_tabular('StdBonusSum', np.std(bonus_sums))
            if self.algo.batch_aggregate_n > 1:
                new_bonuses = tensor_utils.concat_tensor_list(
                            [path["bonuses"] for path in self.experience_replay[-1]]
                            )
                logger.record_tabular('NewPathsMeanBonus',np.mean(new_bonuses))
                logger.record_tabular('NewPathsStdBonus',np.std(new_bonuses))
                logger.record_tabular('NewPathsMaxBonus',np.max(new_bonuses))

        if self.algo.safety_constraint:
            logger.record_tabular('SafetyEval',samples_data['safety_eval'])
            """log the true, raw, undiscounted returns, regardless of what we optimize for"""
            safety_returns = np.array([np.sum(path["safety_rewards"]) for path in paths])
            logger.record_tabular('MeanSafety[U]Return', np.mean(safety_returns))
            logger.record_tabular('StdSafety[U]Return', np.std(safety_returns))
            logger.record_tabular('MaxSafety[U]Return', np.max(safety_returns))

            if self.algo.batch_aggregate_n > 1:
                new_safety_returns = np.array([np.sum(path["safety_rewards"]) for path in self.experience_replay[-1]])
                logger.record_tabular('NewPathsMeanSafety[U]Return',np.mean(new_safety_returns))
                logger.record_tabular('NewPathsStdSafety[U]Return',np.std(new_safety_returns))
                logger.record_tabular('NewPathsMaxSafety[U]Return',np.max(new_safety_returns))

            if self.use_safety_bonus:
                safety_robust_returns = np.array([np.sum(path["safety_robust_rewards"]) for path in paths])
                logger.record_tabular('MeanRobustSafety[U]Return', np.mean(safety_robust_returns))
                logger.record_tabular('StdRobustSafety[U]Return', np.std(safety_robust_returns))
                logger.record_tabular('MaxRobustSafety[U]Return', np.max(safety_robust_returns))

                if self.algo.batch_aggregate_n > 1:
                    new_safety_robust_returns = np.array([np.sum(path["safety_robust_rewards"]) for path in self.experience_replay[-1]])
                    logger.record_tabular('NewPathsMeanRobustSafety[U]Return', np.mean(new_safety_robust_returns))
                    logger.record_tabular('NewPathsStdRobustSafety[U]Return', np.std(new_safety_robust_returns))
                    logger.record_tabular('NewPathsMaxRobustSafety[U]Return', np.max(new_safety_robust_returns))


    def get_IS(self,age):
        if self.algo.decision_weight_mode=='pd':
            return tensor_utils.concat_tensor_list(self.IS_coeffs[age])
        else:
            return np.array(self.IS_coeffs[age])


    def update_parametrized_models(self):
        """only most recent batch of data is used to fit models"""

        logger.log("fitting objective baseline with target_key=" + self.algo.baseline._target_key + "...")
        self.algo.baseline.fit(self.experience_replay[-1])
        logger.log("fitted")

        if self.algo.exploration_bonus:
            logger.log("fitting exploration bonus model...")
            self.algo.exploration_bonus.fit(self.experience_replay[-1])
            logger.log("fitted")

        if self.algo.safety_constraint:
            logger.log("fitting safety constraint model...")
            self.algo.safety_constraint.fit(self.experience_replay[-1])
            logger.log("fitted")
