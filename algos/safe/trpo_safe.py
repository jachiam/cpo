from sandbox.cpo.algos.safe.policy_gradient_safe import PolicyGradientSafe
from sandbox.cpo.optimizers.conjugate_constraint_optimizer import ConjugateConstraintOptimizer
from sandbox.cpo.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from rllab.core.serializable import Serializable
from sandbox.cpo.safety_constraints.trivial import TrivialSafetyConstraint


class TRPO(PolicyGradientSafe, Serializable):
    """
    Trust Region Policy Optimization
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            safety_constrained_optimizer=True,
            safety_constraint=None,
            **kwargs):
        Serializable.quick_init(self, locals())
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            if safety_constraint is not None and safety_constrained_optimizer:
                optimizer = ConjugateConstraintOptimizer(**optimizer_args)
            else:
                optimizer = ConjugateGradientOptimizer(**optimizer_args)
        super(TRPO, self).__init__(optimizer=optimizer, 
                                   safety_constrained_optimizer=safety_constrained_optimizer,
                                   safety_constraint=safety_constraint,
                                   **kwargs)
