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
from typing import Optional, List, Dict, Tuple
import numpy as np
import autograd.numpy as anp

from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.likelihood import (
    MarginalLikelihood,
    GaussianProcessMarginalLikelihood,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.independent.posterior_state import (
    IndependentGPPerResourcePosteriorState,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.constants import (
    INITIAL_NOISE_VARIANCE,
    NOISE_VARIANCE_LOWER_BOUND,
    NOISE_VARIANCE_UPPER_BOUND,
    DEFAULT_ENCODING,
    INITIAL_COVARIANCE_SCALE,
    COVARIANCE_SCALE_LOWER_BOUND,
    COVARIANCE_SCALE_UPPER_BOUND,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.distribution import (
    Gamma,
    LogNormal,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.gluon_blocks_helpers import (
    encode_unwrap_parameter,
    register_parameter,
    create_encoding,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.kernel import (
    KernelFunction,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.mean import (
    MeanFunction,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.posterior_state import (
    PosteriorState,
)
from syne_tune.optimizer.schedulers.utils.simple_profiler import (
    SimpleProfiler,
)


class IndependentGPPerResourceMarginalLikelihood(MarginalLikelihood):
    """
    Marginal likelihood for GP multi-fidelity model over f(x, r), where for
    each r, f(x, r) is represented by an independent GP. The different
    processes share the same kernel, but have their own mean functions mu_r
    and covariance scales c_r.
    """

    def __init__(
        self,
        kernel: KernelFunction,
        mean: Dict[int, MeanFunction],
        resource_attr_range: Tuple[int, int],
        initial_noise_variance=None,
        initial_covariance_scale=None,
        encoding_type=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if initial_noise_variance is None:
            initial_noise_variance = INITIAL_NOISE_VARIANCE
        if initial_covariance_scale is None:
            initial_covariance_scale = INITIAL_COVARIANCE_SCALE
        if encoding_type is None:
            encoding_type = DEFAULT_ENCODING
        self.encoding_noise = create_encoding(
            encoding_name=encoding_type,
            init_val=initial_noise_variance,
            constr_lower=NOISE_VARIANCE_LOWER_BOUND,
            constr_upper=NOISE_VARIANCE_UPPER_BOUND,
            dimension=1,
            prior=Gamma(mean=0.1, alpha=0.1),
        )
        self.encoding_covscale = create_encoding(
            encoding_name=encoding_type,
            init_val=initial_covariance_scale,
            constr_lower=COVARIANCE_SCALE_LOWER_BOUND,
            constr_upper=COVARIANCE_SCALE_UPPER_BOUND,
            dimension=1,
            prior=LogNormal(0.0, 1.0),
        )
        self.mean = mean.copy()
        for v in self.mean.values():
            self.register_child(v)
        self.kernel = kernel
        self.resource_attr_range = resource_attr_range
        with self.name_scope():
            self.noise_variance_internal = register_parameter(
                self.params, "noise_variance", self.encoding_noise
            )
            self.covariance_scale_internal = {
                resource: register_parameter(
                    self.params, f"covariance_scale_{resource}", self.encoding_covscale
                )
                for resource in mean.keys()
            }

    def _noise_variance(self):
        return encode_unwrap_parameter(
            self.noise_variance_internal, self.encoding_noise
        )

    def _covariance_scale(self):
        return {
            resource: encode_unwrap_parameter(internal, self.encoding_covscale)
            for resource, internal in self.covariance_scale_internal.items()
        }

    def get_posterior_state(self, data: dict) -> PosteriorState:
        GaussianProcessMarginalLikelihood._assert_data_entries(data)
        return IndependentGPPerResourcePosteriorState(
            features=data["features"],
            targets=data["targets"],
            kernel=self.kernel,
            mean=self.mean,
            covariance_scale=self._covariance_scale(),
            noise_variance=self._noise_variance(),
            resource_attr_range=self.resource_attr_range,
        )

    def forward(self, data: dict):
        return self.get_posterior_state(data).neg_log_likelihood()

    def param_encoding_pairs(self) -> List[tuple]:
        result = [
            (self.noise_variance_internal, self.encoding_noise)
        ] + self.kernel.param_encoding_pairs()
        for mean in self.mean.values():
            result.extend(mean.param_encoding_pairs())
        for internal in self.covariance_scale_internal.values():
            result.append((internal, self.encoding_covscale))
        return result

    def get_noise_variance(self, as_ndarray=False):
        noise_variance = self._noise_variance()
        return noise_variance if as_ndarray else anp.reshape(noise_variance, (1,))[0]

    def set_noise_variance(self, val: float):
        self.encoding_noise.set(self.noise_variance_internal, val)

    def get_covariance_scale(self, resource: int, as_ndarray=False):
        cov_scale = self._covariance_scale()[resource]
        return cov_scale if as_ndarray else anp.reshape(cov_scale, (1,))[0]

    def set_covariance_scale(self, resource: int, val: float):
        internal = self.covariance_scale_internal.get(resource)
        assert internal is not None, f"resource = {resource} not supported"
        self.encoding_covscale.set(internal, val)

    def _components(self):
        return [("kernel_", self.kernel)] + [
            (f"mean{resource}_", mean) for resource, mean in self.mean.items()
        ]

    def get_params(self) -> Dict[str, np.ndarray]:
        result = {"noise_variance": self.get_noise_variance()}
        for pref, func in self._components():
            result.update({(pref + k): v for k, v in func.get_params().items()})
        for resource, val in self._covariance_scale().items():
            result[f"covariance_scale{resource}"] = val
        return result

    def set_params(self, param_dict: Dict[str, np.ndarray]):
        for pref, func in self._components():
            len_pref = len(pref)
            stripped_dict = {
                k[len_pref:]: v for k, v in param_dict.items() if k.startswith(pref)
            }
            func.set_params(stripped_dict)
        for resource in self.covariance_scale_internal.keys():
            self.set_covariance_scale(
                resource, param_dict[f"covariance_scale{resource}"]
            )
        self.set_noise_variance(param_dict["noise_variance"])

    def on_fit_start(self, data: dict, profiler: Optional[SimpleProfiler] = None):
        GaussianProcessMarginalLikelihood._assert_data_entries(data)
        targets = data["targets"]
        assert (
            targets.shape[1] == 1
        ), "targets cannot be a matrix if parameters are to be fit"
