from rllab.core.parameterized import Parameterized
from rllab.core.serializable import Serializable
import rllab.misc.logger as logger


class SafetyConstraint(object):

    def __init__(self, max_value=1., baseline=None, **kwargs):
        self.max_value = max_value
        self.has_baseline = baseline is not None
        if self.has_baseline:
            self.baseline = baseline

    def evaluate(self, paths):
        raise NotImplementedError

    def fit(self, paths):
        if self.has_baseline:
            logger.log("fitting safety baseline using target_key=" + self.baseline._target_key + "...")
            self.baseline.fit(paths)
            logger.log("fitted")

    def get_safety_step(self):
        return self.max_value

