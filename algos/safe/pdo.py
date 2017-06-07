from sandbox.cpo.algos.safe.policy_gradient_safe import PolicyGradientSafe
from sandbox.cpo.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from rllab.core.serializable import Serializable
import rllab.misc.logger as logger


class PDO(PolicyGradientSafe, Serializable):
    """
    Primal-Dual Optimization
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            safety_constraint=None,
            pdo_vf_mode=1,
            **kwargs):
        Serializable.quick_init(self, locals())
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = ConjugateGradientOptimizer(**optimizer_args)

        pop_keys = ['safety_constrained_optimizer',
                    'safety_tradeoff',
                    'learn_safety_tradeoff_coeff',
                    'safety_key']

        for key in pop_keys:
            if key in kwargs.keys():
                kwargs.pop(key)

        if pdo_vf_mode == 1:
            # won't be using safety baseline, so key should not be advantages.
            safety_key = 'returns'
        else:
            safety_key = 'advantages'

        if pdo_vf_mode == 2 and not(hasattr(safety_constraint,'baseline')):
            logger.log("Warning: selected two-VF PDO, without providing VF for safety constraint.")
            logger.log("Defaulting to one-VF PDO.")
            pdo_vf_mode = 1
            safety_key = 'returns'

        super(PDO, self).__init__(optimizer=optimizer, 
                                   safety_constrained_optimizer=False,
                                   safety_constraint=safety_constraint,
                                   safety_tradeoff=True,
                                   learn_safety_tradeoff_coeff=True,
                                   safety_key=safety_key,
                                   pdo_vf_mode=pdo_vf_mode,
                                   **kwargs)
