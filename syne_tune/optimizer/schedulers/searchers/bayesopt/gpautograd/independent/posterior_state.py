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
from typing import Dict, Tuple, Optional
import numpy as np
import autograd.numpy as anp
from numpy.random import RandomState

from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.posterior_state import (
    PosteriorState,
    GaussProcPosteriorState,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.hp_ranges_impl import (
    decode_extended_features,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.kernel import (
    KernelFunction,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.mean import (
    MeanFunction,
)


class IndependentGPPerResourcePosteriorState(PosteriorState):
    """
    Posterior state for model over f(x, r), where for a fixed set of resource
    levels r, each f(x, r) is represented by an independent Gaussian process.
    These processes share a common covariance function k(x, x), but can have
    their own mean functions mu_r and covariance scales c_r.

    Attention: Predictions can only be done at (x, r) where r has at least
    one training datapoint. This is because a posterior state cannot
    represent the prior.
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        kernel: KernelFunction,
        mean: Dict[int, MeanFunction],
        covariance_scale: Dict[int, np.ndarray],
        noise_variance: np.ndarray,
        resource_attr_range: Tuple[int, int],
        debug_log: bool = False,
    ):
        """
        `mean` and `covariance_scale` map supported resource levels r to
        mean function mu_r and covariance scale c_r.

        :param features: Input points X, extended features, shape (n, d)
        :param targets: Targets Y, shape (n, m)
        :param kernel: Kernel function k(X, X')
        :param mean: See above
        :param covariance_scale: See above
        :param noise_variance: Noise variance sigsq, shape (1,)
        :param resource_attr_range: (r_min, r_max)
        """
        assert isinstance(kernel, KernelFunction), "kernel must be KernelFunction"
        assert set(mean.keys()) == set(
            covariance_scale.keys()
        ), "mean, covariance_scale must have the same keys"
        self._compute_states(
            features,
            targets,
            kernel,
            mean,
            covariance_scale,
            noise_variance,
            resource_attr_range,
            debug_log,
        )
        self._num_data = features.shape[0]
        self._num_features = features.shape[1]
        self._num_fantasies = targets.shape[1]
        self._resource_attr_range = resource_attr_range

    def _compute_states(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        kernel: KernelFunction,
        mean: Dict[int, MeanFunction],
        covariance_scale: Dict[int, np.ndarray],
        noise_variance: np.ndarray,
        resource_attr_range: Tuple[int, int],
        debug_log: bool = False,
    ):
        features, resources = decode_extended_features(features, resource_attr_range)
        self._states = dict()
        for resource, mean_function in mean.items():
            cov_scale = covariance_scale[resource]
            rows = np.flatnonzero(resources == resource)
            if rows.size > 0:
                r_features = features[rows]
                r_targets = targets[rows]
                self._states[resource] = GaussProcPosteriorState(
                    features=r_features,
                    targets=r_targets,
                    mean=mean_function,
                    kernel=(kernel, cov_scale),
                    noise_variance=noise_variance,
                    debug_log=debug_log,
                )

    @property
    def num_data(self):
        return self._num_data

    @property
    def num_features(self):
        return self._num_features

    @property
    def num_fantasies(self):
        return self._num_fantasies

    def neg_log_likelihood(self) -> anp.ndarray:
        return anp.sum([state.neg_log_likelihood() for state in self._states.values()])

    def predict(self, test_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        features_per_resource = self._split_features(test_features)
        num_test = test_features.shape[0]
        posterior_means = np.zeros((num_test, self.num_fantasies))
        posterior_variances = np.zeros((num_test,))
        for resource, (features, rows) in features_per_resource.items():
            p_means, p_vars = self._states[resource].predict(features)
            assert (
                p_means.ndim == 2 and p_means.shape[1] == self.num_fantasies
            ), f"r={resource}, p_means.shape = {p_means.shape}"
            posterior_means[rows] = p_means
            posterior_variances[rows] = p_vars
        return posterior_means, posterior_variances

    def sample_marginals(
        self,
        test_features: np.ndarray,
        num_samples: int = 1,
        random_state: Optional[RandomState] = None,
    ) -> np.ndarray:
        features_per_resource = self._split_features(test_features)
        num_test = test_features.shape[0]
        nf = self.num_fantasies
        shp = (num_test, num_samples) if nf == 1 else (num_test, nf, num_samples)
        samples = np.zeros(shp)
        for resource, (features, rows) in features_per_resource.items():
            s_margs = self._states[resource].sample_marginals(
                features, num_samples, random_state
            )
            samples[rows] = s_margs
        return samples

    def _split_features(self, features: np.ndarray):
        features, resources = decode_extended_features(
            features, self._resource_attr_range
        )
        result = dict()
        for resource in set(resources):
            assert resource in self._states, (
                f"Resource r = {resource} is not covered by data in posterior"
                + " state"
            )
            rows = np.flatnonzero(resources == resource)
            result[resource] = (features[rows], rows)
        return result

    def backward_gradient(
        self,
        input: np.ndarray,
        head_gradients: Dict[str, np.ndarray],
        mean_data: float,
        std_data: float,
    ) -> np.ndarray:
        input, resource = decode_extended_features(
            input.reshape((1, -1)), self._resource_attr_range
        )
        assert len(resource) == 1
        resource = resource[0]
        return self._states[resource].backward_gradient(
            input, head_gradients, mean_data, std_data
        )

    def sample_joint(
        self,
        test_features: np.ndarray,
        num_samples: int = 1,
        random_state: Optional[RandomState] = None,
    ) -> np.ndarray:
        features_per_resource = self._split_features(test_features)
        num_test = test_features.shape[0]
        nf = self.num_fantasies
        shp = (num_test, num_samples) if nf == 1 else (num_test, nf, num_samples)
        samples = np.zeros(shp)
        for resource, (features, rows) in features_per_resource.items():
            s_joint = self._states[resource].sample_joint(
                features, num_samples, random_state
            )
            samples[rows] = s_joint
        return samples
