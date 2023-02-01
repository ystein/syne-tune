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
from typing import List, Dict, Any

from syne_tune.optimizer.schedulers.hyperband_stopping import (
    RungSystem,
)
from syne_tune.optimizer.schedulers.searchers.dyhpo.dyhpo_searcher import (
    DynamicHPOSearcher,
)


class DyHPORungSystem(RungSystem):
    """
    Implements the logic which decides which paused trial to promote to the
    next resource level, or alternatively which configuration to start as a
    new trial, proposed in:

        | Wistuba, M. and Kadra, A. and Grabocka, J.
        | Dynamic and Efficient Gray-Box Hyperparameter Optimization for Deep Learning
        | https://arxiv.org/abs/2202.09774

    We do promotion-based scheduling, as in
    :class:`~syne_tune.optimizer.schedulers.hyperband_promotion.PromotionRungSystem`.
    All relevant logic happens in :meth:`on_task_schedule`. There, we score all
    paused trials as well as a number of new configurations. If a new
    configuration wins, this is passed back instead of a trial ID to be promoted.

    We can ignore ``rung_levels`` and ``promote_quantiles``, they are not used.
    For each trial, we only need to maintain the resource level at which it is
    paused.
    """

    def __init__(
        self,
        rung_levels: List[int],
        promote_quantiles: List[float],
        metric: str,
        mode: str,
        resource_attr: str,
        max_t: int,
        searcher: DynamicHPOSearcher,
    ):
        super().__init__(
            rung_levels, promote_quantiles, metric, mode, resource_attr, max_t
        )
        # Note: We do not use ``_rungs`` as created by the super-class
        self._searcher = searcher
        self._min_resource = rung_levels[0]
        # For each paused trial, ``_paused_trial`` maps ``trial_id`` to
        # its current resource level. Once a trial reaches ``max_t``, it is
        # removed there
        self._paused_trial = dict()

    def on_task_schedule(self) -> Dict[str, Any]:
        """
        The main decision making happens here. We collect ``(trial_id, resource)``
        for all paused trials and call ``searcher``. The searcher scores all
        these trials along with a certain number of randomly drawn new
        configurations.

        If one of the paused trials has the best score, we return its ``trial_id``
        along with extra information, so it gets promoted.
        If one of the new configurations has the best score, we return this
        configuration (key "new_configuration"). In this case, a new trial is
        started with this configuration.
        """
        paused_trials = [
            (trial_id, resource) for trial_id, resource in self._paused_trial.items()
        ]
        result = self._searcher.score_paused_trials_and_new_configs(paused_trials)
        trial_id = result.get("trial_id")
        if trial_id is not None:
            # Trial is to be promoted
            resource = self._paused_trial.get(trial_id)
            assert resource is not None  # Sanity check
            ret_dict = {
                "trial_id": trial_id,
                "resume_from": resource,
                "milestone": resource + 1,
            }
        else:
            # New trial is to be started
            ret_dict = {"new_configuration": result["config"]}
        return ret_dict

    def on_task_report(
        self, trial_id: str, result: Dict[str, Any], skip_rungs: int
    ) -> Dict[str, Any]:
        resource = int(result[self._resource_attr])
        # This default is returned if ``resource`` is not larger than what we
        # saw before from trial. This happens if checkpointing is not
        # implemented.
        milestone_reached = False
        next_milestone = None
        ignore_data = True
        if resource >= self._min_resource:
            current_resource = self._paused_trial.get(trial_id)
            if current_resource is None or resource > current_resource:
                milestone_reached = True
                ignore_data = False
                if resource < self._max_t:
                    self._paused_trial[trial_id] = resource
                    next_milestone = resource + 1
                elif current_resource is not None:
                    del self._paused_trial[trial_id]
        return {
            "task_continues": not milestone_reached,
            "milestone_reached": milestone_reached,
            "next_milestone": next_milestone,
            "ignore_data": ignore_data,
        }

    @staticmethod
    def does_pause_resume() -> bool:
        return True