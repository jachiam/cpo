from sandbox.cpo.algos.safe.policy_gradient_safe import PolicyGradientSafe
from sandbox.cpo.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from rllab.core.serializable import Serializable
import rllab.misc.logger as logger


class FPO(PolicyGradientSafe, Serializable):
    """
    Fixed Penalty Optimization
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            safety_constraint=None,
            **kwargs):
        Serializable.quick_init(self, locals())
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = ConjugateGradientOptimizer(**optimizer_args)

        pop_keys = ['safety_constrained_optimizer',
                    'safety_tradeoff',
                    'learn_safety_tradeoff_coeff',
                    'safety_key',
                    'pdo_vf_mode']

        for key in pop_keys:
            if key in kwargs.keys():
                kwargs.pop(key)

        safety_key = 'returns'
        pdo_vf_mode = 1

        super(FPO, self).__init__(optimizer=optimizer, 
                                   safety_constrained_optimizer=False,
                                   safety_constraint=safety_constraint,
                                   safety_tradeoff=True,
                                   learn_safety_tradeoff_coeff=False,
                                   safety_key=safety_key,
                                   pdo_vf_mode=pdo_vf_mode,
                                   **kwargs)
