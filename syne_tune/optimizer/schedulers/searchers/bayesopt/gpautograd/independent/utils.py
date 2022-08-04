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
from typing import Optional, Tuple, List
from numpy.random import RandomState

from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.tuning_job_state import (
    TuningJobState,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common import (
    Configuration,
    INTERNAL_METRIC_NAME,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.hp_ranges import (
    HyperparameterRanges,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.posterior_state import (
    PosteriorStateWithSampleJoint,
    GaussProcPosteriorState,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.independent.posterior_state import (
    IndependentGPPerResourcePosteriorState,
)


def hypertune_ranking_losses(
    poster_state: IndependentGPPerResourcePosteriorState,
    data: TuningJobState,
    num_samples: int,
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
    data_max_resource = _extract_data_at_resource(data, max_resource)
    assert len(data_max_resource) < 6, (
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
            hp_ranges=data.hp_ranges,
            data_max_resource=data_max_resource,
            num_samples=num_samples,
            random_state=random_state,
        )
    if num_brackets == num_rungs:
        # Loss values for maximum rung: Five-fold cross-validation
        loss_values[-1] = _losses_for_maximum_rung_by_cross_validation(
            poster_state=poster_state,
            hp_ranges=data.hp_ranges,
            data_max_resource=data_max_resource,
            num_samples=num_samples,
            random_state=random_state,
        )
    return loss_values


def _extract_data_at_resource(
    data: TuningJobState, resource: int
) -> List[Tuple[Configuration, float]]:
    result = []
    targets = []
    resource = str(resource)
    for trial_evaluation in data.trials_evaluations:
        metric_vals = trial_evaluation.metrics.get(INTERNAL_METRIC_NAME)
        if metric_vals is not None:
            assert isinstance(metric_vals, dict)
            metric_val = metric_vals.get(resource)
            if metric_val is not None:
                result.append(
                    (data.config_for_trial[trial_evaluation.trial_id], metric_val)
                )
                targets.append(metric_val)
    # Normalize target values
    mn = np.mean(targets).item()
    sd = max(np.std(targets).item(), 1e-9)
    return [(config, (target - mn) / sd) for config, target in result]


def _features_targets_from_data(
    data: List[Tuple[Configuration, float]],
    hp_ranges: HyperparameterRanges,
) -> (np.ndarray, np.ndarray):
    return (
        hp_ranges.to_ndarray_matrix([x[0] for x in data]),
        np.array([x[1] for x in data]),
    )


def _losses_for_maximum_rung_by_cross_validation(
    poster_state: IndependentGPPerResourcePosteriorState,
    hp_ranges: HyperparameterRanges,
    data_max_resource: List[Tuple[Configuration, float]],
    num_samples: int,
    random_state: Optional[RandomState] = None,
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
    num_data = len(data_max_resource)
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
        train_data = data_max_resource[:start] + data_max_resource[end:]
        test_data = data_max_resource[start:end]
        start = end
        # Note: If there are pending evaluations at the highest level, they
        # are not taken into account here (no fantasizing).
        features, targets = _features_targets_from_data(
            data=train_data, hp_ranges=hp_ranges
        )
        poster_state_fold = GaussProcPosteriorState(
            features=features,
            targets=targets,
            mean=poster_state_max_resource.mean,
            kernel=poster_state_max_resource.kernel,
            noise_variance=poster_state_max_resource.noise_variance,
            debug_log=poster_state_max_resource.debug_log,
        )
        result += _losses_for_rung(
            resource=max_resource,
            poster_state=poster_state_fold,
            hp_ranges=hp_ranges,
            data_max_resource=test_data,
            num_samples=num_samples,
            random_state=random_state,
        )
    result *= 1 / num_folds
    return result


def _losses_for_rung(
    resource: int,
    poster_state: PosteriorStateWithSampleJoint,
    hp_ranges: HyperparameterRanges,
    data_max_resource: List[Tuple[Configuration, float]],
    num_samples: int,
    random_state: Optional[RandomState] = None,
) -> np.ndarray:
    num_data = len(data_max_resource)
    result = np.zeros(num_samples)
    for j, k in ((j, k) for j in range(num_data - 1) for k in range(j + 1, num_data)):
        yj_sub_yk = data_max_resource[j][1] - data_max_resource[k][1]
        fj_sub_fk = _sample_pair_function_values(
            poster_state=poster_state,
            hp_ranges=hp_ranges,
            config1=data_max_resource[j][0],
            config2=data_max_resource[k][0],
            resource=resource,
            num_samples=num_samples,
            random_state=random_state,
        )
        assert fj_sub_fk.size == num_samples, fj_sub_fk.shape
        result += yj_sub_yk * fj_sub_fk.reshape((-1,)) < 0
    return result


def _sample_pair_function_values(
    poster_state: PosteriorStateWithSampleJoint,
    hp_ranges: HyperparameterRanges,
    config1: Configuration,
    config2: Configuration,
    resource: int,
    num_samples: int,
    random_state: Optional[RandomState] = None,
) -> np.ndarray:
    if isinstance(poster_state, IndependentGPPerResourcePosteriorState):
        resource_attr_name = hp_ranges.name_last_pos
        extra_dct = {resource_attr_name: resource}
    else:
        extra_dct = dict()
    features = hp_ranges.to_ndarray_matrix(
        [dict(config1, **extra_dct), dict(config2, **extra_dct)]
    )
    sample = poster_state.sample_joint(
        test_features=features, num_samples=num_samples, random_state=random_state
    )
    result = sample[0] - sample[1]
    if poster_state.num_fantasies > 1:
        result = np.mean(result, axis=0)  # Average over fantasies
    return result
