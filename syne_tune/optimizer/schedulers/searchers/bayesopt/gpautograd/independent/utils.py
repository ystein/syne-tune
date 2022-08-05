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
import numpy as np
from typing import Optional, Tuple
from numpy.random import RandomState

from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.hp_ranges_impl import (
    decode_extended_features,
    HyperparameterRangeInteger,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.scaling import (
    LinearScaling,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.posterior_state import (
    PosteriorStateWithSampleJoint,
    GaussProcPosteriorState,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.independent.posterior_state import (
    IndependentGPPerResourcePosteriorState,
)

__all__ = ["hypertune_ranking_losses", "extract_data_at_resource"]


def hypertune_ranking_losses(
    poster_state: IndependentGPPerResourcePosteriorState,
    data: dict,
    num_samples: int,
    resource_attr_range: Tuple[int, int],
    num_brackets: Optional[int] = None,
    random_state: Optional[RandomState] = None,
) -> np.ndarray:
    """
    Samples ranking loss values as defined in the Hyper-Tune paper. We return a
    matrix of size `(num_brackets, num_samples)`. If `num_brackets` is not given,
    it is set to the number of rungs `num_rungs = poster_state.rung_levels`.
    The loss values depend on the cases in `data` at the highest level
    `poster_state.rung_levels[-1]`, there must be at least 6 of those.

    If `num_brackets == num_rungs`, loss values at the highest level are
    estimated by cross-validation (so the data at the highest level is split
    into training and test, where the training part is used to obtain the
    posterior state). The number of CV folds is `<= 5`, and such that each fold
    has at least two points.

    :param poster_state: Posterior state (independent across levels)
    :param data: Training data (only data at highest level is used)
    :param num_samples: Number of independent loss samples
    :param resource_attr_range: (r_min, r_max)
    :param num_brackets: See above
    :param random_state: PRNG state
    :return: See above
    """
    rung_levels = poster_state.rung_levels
    num_rungs = len(rung_levels)
    if num_brackets is None:
        num_brackets = num_rungs
    else:
        assert (
            2 <= num_brackets <= num_rungs
        ), f"num_brackets = {num_brackets}, must be in [2, {num_rungs}]"
    max_resource = rung_levels[-1]
    data_max_resource = extract_data_at_resource(
        data=data, resource=max_resource, resource_attr_range=resource_attr_range
    )
    assert data_max_resource["features"].shape[0] < 6, (
        "Must have at least six observed datapoints at maximum resource "
        f"level {max_resource}"
    )
    loss_values = np.zeros((num_brackets, num_samples))
    # All loss values except for maximum rung (which is special)
    end = min(num_brackets, num_rungs - 1)
    for pos, resource in enumerate(rung_levels[:end]):
        loss_values[pos] = _losses_for_rung(
            resource=resource,
            poster_state=poster_state,
            data_max_resource=data_max_resource,
            num_samples=num_samples,
            resource_attr_range=resource_attr_range,
            random_state=random_state,
        )
    if num_brackets == num_rungs:
        # Loss values for maximum rung: Five-fold cross-validation
        loss_values[-1] = _losses_for_maximum_rung_by_cross_validation(
            poster_state=poster_state,
            data_max_resource=data_max_resource,
            num_samples=num_samples,
            resource_attr_range=resource_attr_range,
            random_state=random_state,
        )
    return loss_values


def extract_data_at_resource(
    data: dict, resource: int, resource_attr_range: Tuple[int, int]
) -> dict:
    features_ext = data["features"]
    targets = data["targets"]
    num_fantasies = targets.shape[1] if targets.ndim == 2 else 1
    resource = str(resource)
    features, resources = decode_extended_features(
        features_ext=features_ext, resource_attr_range=resource_attr_range
    )
    ind = resources == resource
    features_max = features[ind]
    targets_max = targets[ind].reshape((-1, num_fantasies))
    if num_fantasies > 1:
        # Remove pending evaluations at highest level (they are ignored). We
        # detect observed cases by all target values being the same.
        ind = np.array([x == np.full(num_fantasies, x[0]) for x in targets_max])
        features_max = features_max[ind]
        targets_max = targets_max[ind, 0]
    return {"features": features_max, "targets": targets_max.reshape((-1,))}


def _losses_for_maximum_rung_by_cross_validation(
    poster_state: IndependentGPPerResourcePosteriorState,
    data_max_resource: dict,
    num_samples: int,
    resource_attr_range: Tuple[int, int],
    random_state: Optional[RandomState],
) -> np.ndarray:
    """
    Estimates loss samples at highest rung by K-fold cross-validation, where
    `K <= 5` is chosen such that each fold has at least 2 points (since
    `len(data_max_resource) >= 6`, we have `K >= 3`).

    For simplicity, we ignore pending evaluations here. They would affect the
    result only if they are at the highest level.
    """
    rung_levels = poster_state.rung_levels
    max_resource = rung_levels[-1]
    poster_state_max_resource = poster_state.state(max_resource)
    features = data_max_resource["features"]
    targets = data_max_resource["targets"]
    num_data = features.shape[0]
    # Make sure each fold has at least two datapoints
    num_folds = min(num_data // 2, 5)
    fold_sizes = np.full(num_folds, num_data // num_folds)
    incr_ind = num_folds - num_data + (num_data // num_folds) * num_folds
    fold_sizes[incr_ind:] += 1
    # Loop over folds
    result = np.zeros(num_samples)
    start = 0
    for fold_size in fold_sizes:
        end = start + fold_size
        train_data = {
            "features": np.vstack((features[:start], features[end:])),
            "targets": np.concatenate((targets[:start], targets[end:])),
        }
        test_data = {
            "features": features[start:end],
            "targets": targets[start:end],
        }
        start = end
        # Note: If there are pending evaluations at the highest level, they
        # are not taken into account here (no fantasizing).
        poster_state_fold = GaussProcPosteriorState(
            **train_data,
            mean=poster_state_max_resource.mean,
            kernel=poster_state_max_resource.kernel,
            noise_variance=poster_state_max_resource.noise_variance,
        )
        result += _losses_for_rung(
            resource=max_resource,
            poster_state=poster_state_fold,
            data_max_resource=test_data,
            num_samples=num_samples,
            resource_attr_range=resource_attr_range,
            random_state=random_state,
        )
    result *= 1 / num_folds
    return result


def _losses_for_rung(
    resource: int,
    poster_state: PosteriorStateWithSampleJoint,
    data_max_resource: dict,
    num_samples: int,
    resource_attr_range: Tuple[int, int],
    random_state: Optional[RandomState] = None,
) -> np.ndarray:
    num_data = len(data_max_resource)
    result = np.zeros(num_samples)
    hp_range = HyperparameterRangeInteger(
        name="resource",
        lower_bound=resource_attr_range[0],
        upper_bound=resource_attr_range[1],
        scaling=LinearScaling(),
    )
    resource_encoded = hp_range.to_ndarray(resource)
    features = data_max_resource["features"]
    targets = data_max_resource["targets"]
    for j, k in ((j, k) for j in range(num_data - 1) for k in range(j + 1, num_data)):
        yj_sub_yk = targets[j] - targets[k]
        fj_sub_fk = _sample_pair_function_values(
            poster_state=poster_state,
            features1=features[j],
            features2=features[k],
            resource_encoded=resource_encoded,
            num_samples=num_samples,
            random_state=random_state,
        )
        assert fj_sub_fk.size == num_samples, fj_sub_fk.shape
        result += yj_sub_yk * fj_sub_fk.reshape((-1,)) < 0
    return result


def _sample_pair_function_values(
    poster_state: PosteriorStateWithSampleJoint,
    features1: np.ndarray,
    features2: np.ndarray,
    resource_encoded: np.ndarray,
    num_samples: int,
    random_state: Optional[RandomState],
) -> np.ndarray:
    features1 = np.reshape(features1, (1, -1))
    features2 = np.reshape(features2, (1, -1))
    if isinstance(poster_state, IndependentGPPerResourcePosteriorState):
        resource_encoded = np.reshape(resource_encoded, (1, 1))
        features1 = np.concatenate((features1, resource_encoded), axis=1)
        features2 = np.concatenate((features2, resource_encoded), axis=1)
    features = np.concatenate((features1, features2), axis=0)
    sample = poster_state.sample_joint(
        test_features=features, num_samples=num_samples, random_state=random_state
    )
    result = sample[0] - sample[1]
    if poster_state.num_fantasies > 1:
        result = np.mean(result, axis=0)  # Average over fantasies
    return result
