# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
import logging
from typing import Optional, List

from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.constants import (
    OptimizationConfig,
    DEFAULT_OPTIMIZATION_CONFIG,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.gp_model import (
    GaussianProcessModel,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.kernel import (
    KernelFunction,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.likelihood import (
    GaussianProcessMarginalLikelihood,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.mean import (
    ScalarMeanFunction,
    MeanFunction,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.optimization_utils import (
    apply_lbfgs_with_multiple_starts,
    create_lbfgs_arguments,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.posterior_state import (
    GaussProcPosteriorState,
)
from syne_tune.optimizer.schedulers.utils.simple_profiler import SimpleProfiler

logger = logging.getLogger(__name__)


class GaussianProcessRegression(GaussianProcessModel):
    """
    Gaussian Process Regression

    Takes as input a mean function (which depends on X only) and a kernel
    function.

    :param kernel: Kernel function (for instance, a Matern52---note we cannot
        provide Matern52() as default argument since we need to provide with
        the dimension of points in X)
    :param mean: Mean function (which depends on X only)
    :param initial_noise_variance: Initial value for noise variance parameter
    :param optimization_config: Configuration that specifies the behavior of
        the optimization of the marginal likelihood.
    :param random_seed: Random seed to be used (optional)
    :param fit_reset_params: Reset parameters to initial values before running
        'fit'? If False, 'fit' starts from the current values

    """

    def __init__(
        self,
        kernel: KernelFunction,
        mean: MeanFunction = None,
        initial_noise_variance: float = None,
        optimization_config: OptimizationConfig = None,
        random_seed=None,
        fit_reset_params: bool = True,
    ):
        super().__init__(random_seed)
        if mean is None:
            mean = ScalarMeanFunction()
        if optimization_config is None:
            optimization_config = DEFAULT_OPTIMIZATION_CONFIG
        self._states = None
        self.fit_reset_params = fit_reset_params
        self.optimization_config = optimization_config
        self.likelihood = GaussianProcessMarginalLikelihood(
            kernel=kernel, mean=mean, initial_noise_variance=initial_noise_variance
        )
        self.reset_params()

    @property
    def states(self) -> Optional[List[GaussProcPosteriorState]]:
        return self._states

    def fit(self, data: dict, profiler: SimpleProfiler = None):
        """
        Fit the parameters of the GP by optimizing the marginal likelihood,
        and set posterior states.

        We catch exceptions during the optimization restarts. If any restarts
        fail, log messages are written. If all restarts fail, the current
        parameters are not changed.

        `data['features']` is the data matrix of shape (n, d),
        `data['targets']` the target column vector of shape (n, 1).

        :param data: Input data
        :param profiler: Profiler, optional
        """
        self.likelihood.on_fit_start(data, profiler)
        if self.fit_reset_params:
            self.reset_params()
        n_starts = self.optimization_config.n_starts
        ret_infos = apply_lbfgs_with_multiple_starts(
            *create_lbfgs_arguments(
                criterion=self.likelihood,
                crit_args=[data],
                verbose=self.optimization_config.verbose,
            ),
            bounds=self.likelihood.box_constraints_internal(),
            random_state=self._random_state,
            n_starts=n_starts,
            tol=self.optimization_config.lbfgs_tol,
            maxiter=self.optimization_config.lbfgs_maxiter
        )

        # Logging in response to failures of optimization runs
        n_succeeded = sum(x is None for x in ret_infos)
        if n_succeeded < n_starts:
            log_msg = "[GaussianProcessRegression.fit]\n"
            log_msg += (
                "{} of the {} restarts failed with the following exceptions:\n".format(
                    n_starts - n_succeeded, n_starts
                )
            )
            copy_params = {
                param.name: param.data()
                for param in self.likelihood.collect_params().values()
            }
            for i, ret_info in enumerate(ret_infos):
                if ret_info is not None:
                    log_msg += "- Restart {}: Exception {}\n".format(
                        i, ret_info["type"]
                    )
                    log_msg += "  Message: {}\n".format(ret_info["msg"])
                    log_msg += "  Args: {}\n".format(ret_info["args"])
                    # Set parameters in order to print them. These are the
                    # parameters for which the evaluation failed
                    self._set_likelihood_params(ret_info["params"])
                    log_msg += "  Params: " + str(self.get_params())
                    logger.info(log_msg)
            # Restore parameters
            self._set_likelihood_params(copy_params)
            if n_succeeded == 0:
                logger.info(
                    "All restarts failed: Skipping hyperparameter fitting for now"
                )
        # Recompute posterior state for new hyperparameters
        self._recompute_states(data)

    def _set_likelihood_params(self, params: dict):
        for param in self.likelihood.collect_params().values():
            vec = params.get(param.name)
            if vec is not None:
                param.set_data(vec)

    def recompute_states(self, data: dict):
        """
        We allow targets to be a matrix with m>1 columns, which is useful to support
        fantasizing.

        @param data: See `fit`
        """
        self._recompute_states(data)

    def _recompute_states(self, data: dict):
        self.likelihood.data_precomputations(data)
        self._states = [self.likelihood.get_posterior_state(data)]

    def get_params(self):
        return self.likelihood.get_params()

    def set_params(self, param_dict):
        self.likelihood.set_params(param_dict)

    def reset_params(self):
        """
        Reset hyperparameters to their initial values (or resample them).
        """
        self.likelihood.reset_params(self._random_state)
