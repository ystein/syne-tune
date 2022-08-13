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
from typing import Callable, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass

from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.constants import (
    OptimizationConfig,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.gp_model import (
    GaussianProcessOptimizeModel,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.kernel import (
    KernelFunction,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.likelihood import (
    MarginalLikelihood,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.mean import (
    MeanFunction,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.independent.likelihood import (
    IndependentGPPerResourceMarginalLikelihood,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.independent.posterior_state import (
    IndependentGPPerResourcePosteriorState,
)
from syne_tune.optimizer.schedulers.utils.simple_profiler import SimpleProfiler
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.independent.utils import (
    hypertune_ranking_losses,
    number_supported_brackets_and_data_highest_level,
)

logger = logging.getLogger(__name__)


@dataclass
class HyperTuneDistributionArguments:
    num_samples: int
    num_brackets: Optional[int] = None

    def __post_init__(self):
        assert self.num_brackets is None or self.num_brackets >= 2
        assert self.num_samples >= 1


class IndependentGPPerResourceModel(GaussianProcessOptimizeModel):
    """
    GP multi-fidelity model over f(x, r), where for each r, f(x, r) is
    represented by an independent GP. The different processes share the same
    kernel, but have their own mean functions mu_r and covariance scales c_r.

    The likelihood object is not created at construction, but only with
    `create_likelihood`. This is because we need to know the rung levels of
    the Hyperband scheduler.

    If `hypertune_distribution_args` is given, we also estimate the distribution
    [w_k] over brackets from the Hyper-Tune paper in `fit`. See

        Yang Li et al
        Hyper-Tune: Towards Efficient Hyper-parameter Tuning at Scale
        VLDB 2022

    Our implementation differs from the Hyper-Tune paper in a number of ways.
    Most importantly, their method requires a sufficient number of observed
    points at the starting rung of the highest bracket. In contrast, we
    estimate ranking loss values already when the starting rung of the 2nd
    bracket is sufficiently occupied. This allows us to estimate the head
    of the distribution only (over all brackets with sufficiently occupied
    starting rungs), and we use the default distribution over the remaining
    tail. Eventually, we do the same as Hyper-Tune, but we move away from the
    default distribution earlier on.

    :param kernel: Kernel function without covariance scale, shared by models
        for all resources r
    :param mean_factory: Factory function for mean functions mu_r(x)
    :param resource_attr_range: (r_min, r_max)
    :param initial_noise_variance: Initial value for noise variance parameter
    :param initial_covariance_scale: Initial value for covariance scale
        parameters c_r
    :param optimization_config: Configuration that specifies the behavior of
        the optimization of the marginal likelihood.
    :param random_seed: Random seed to be used (optional)
    :param fit_reset_params: Reset parameters to initial values before running
        'fit'? If False, 'fit' starts from the current values
    :param hypertune_distribution_args: If given, the distribution [w_k] from
        the Hyper-Tune paper is estimated at the end of `fit`
    """

    def __init__(
        self,
        kernel: KernelFunction,
        mean_factory: Callable[[int], MeanFunction],
        resource_attr_range: Tuple[int, int],
        initial_noise_variance: float = None,
        initial_covariance_scale: float = None,
        optimization_config: OptimizationConfig = None,
        random_seed=None,
        fit_reset_params: bool = True,
        hypertune_distribution_args: Optional[HyperTuneDistributionArguments] = None,
    ):
        super().__init__(
            optimization_config=optimization_config,
            random_seed=random_seed,
            fit_reset_params=fit_reset_params,
        )
        self._kernel = kernel
        self._mean_factory = mean_factory
        self._resource_attr_range = resource_attr_range
        self._initial_noise_variance = initial_noise_variance
        self._initial_covariance_scale = initial_covariance_scale
        self.hypertune_distribution_args = hypertune_distribution_args
        self._distribution = None
        # Tuple (num_supp_brackets, data_size) for current distribution. If
        # this signature is different in `fit`, the distribution is recomputed
        self._hypertune_distribution_signature = None
        self._likelihood = None  # Delayed creation

    def create_likelihood(self, rung_levels: List[int]):
        """
        Delayed creation of likelihood, needs to know rung levels of Hyperband
        scheduler.

        Note: last entry of `rung_levels` must be `max_t`, even if this is not
        a rung level in Hyperband.

        :param rung_levels: Rung levels
        """
        mean = {resource: self._mean_factory(resource) for resource in rung_levels}
        self._likelihood = IndependentGPPerResourceMarginalLikelihood(
            kernel=self._kernel,
            mean=mean,
            resource_attr_range=self._resource_attr_range,
            initial_noise_variance=self._initial_noise_variance,
            initial_covariance_scale=self._initial_covariance_scale,
        )
        self.reset_params()

    @property
    def likelihood(self) -> MarginalLikelihood:
        assert (
            self._likelihood is not None
        ), "Call create_likelihood (passing rung levels) in order to complete creation"
        return self._likelihood

    def hypertune_distribution(self) -> Optional[np.ndarray]:
        """
        Distribution of support size `num_supp_brackets`, where
        `num_supp_brackets <= args.num_brackets` (the latter is maximum if
        not given) is maximum such that the first `num_supp_brackets`
        brackets have >= 6 labeled datapoints each.

        If `num_supp_brackets < args.num_brackets`, the distribution must be
        extended to full size before being used to sample the next bracket.
        """
        return self._distribution

    def fit(self, data: dict, profiler: SimpleProfiler = None):
        super().fit(data, profiler)
        args = self.hypertune_distribution_args
        if args is not None:
            poster_state: IndependentGPPerResourcePosteriorState = self.states[0]
            (
                num_supp_brackets,
                data_resource,
            ) = number_supported_brackets_and_data_highest_level(
                rung_levels=poster_state.rung_levels,
                data=data,
                resource_attr_range=self._resource_attr_range,
                num_brackets=args.num_brackets,
            )
            if num_supp_brackets > 1:
                num_data = data_resource["features"].shape[0]
                curr_sig = self._hypertune_distribution_signature
                new_sig = (num_supp_brackets, num_data)
                if curr_sig is None or new_sig != curr_sig:
                    # Data at highest level has changed
                    self._hypertune_distribution_signature = new_sig
                    ranking_losses = hypertune_ranking_losses(
                        poster_state=poster_state,
                        data=data,
                        num_samples=args.num_samples,
                        resource_attr_range=self._resource_attr_range,
                        num_brackets=args.num_brackets,
                        random_state=self.random_state,
                    )
                    min_losses = np.min(ranking_losses, axis=0, keepdims=True)
                    theta = np.sum(ranking_losses == min_losses, axis=1).reshape((-1,))
                    self._distribution = theta * np.array(
                        [1 / r for r in poster_state.rung_levels[: theta.size]]
                    )
                    self._distribution /= np.sum(self._distribution)
