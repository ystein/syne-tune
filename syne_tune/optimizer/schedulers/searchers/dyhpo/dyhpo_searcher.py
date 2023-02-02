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
from typing import Optional, List, Dict, Any, Tuple
import logging
import numpy as np

from syne_tune.optimizer.schedulers.searchers import (
    BaseSearcher,
    GPMultiFidelitySearcher,
)
from syne_tune.optimizer.schedulers.searchers.gp_fifo_searcher import (
    create_initial_candidates_scorer,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.utils.debug_log import (
    DebugLogPrinter,
)
from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.common import (
    ExclusionList,
)
from syne_tune.optimizer.schedulers.searchers.utils.common import (
    Configuration,
)

logger = logging.getLogger(__name__)


INTERNAL_KEY = "RESERVED_KEY_31415927"


class MyGPMultiFidelitySearcher(GPMultiFidelitySearcher):
    """
    This wrapper is for convenience, to avoid having to depend on internal
    concepts of
    :class:`~syne_tune.optimizer.schedulers.searchers.GPMultiFidelitySearcher`.
    """

    def __init__(self, config_space: Dict[str, Any], **kwargs):
        assert (
            INTERNAL_KEY not in config_space
        ), f"Key {INTERNAL_KEY} must not be used in config_space (reserved)"
        super().__init__(config_space, **kwargs)

    def _get_config_modelbased(
        self, exclusion_candidates: ExclusionList, **kwargs
    ) -> Optional[Configuration]:
        return {INTERNAL_KEY: "dummy"}

    def score_paused_trials_and_new_configs(
        self,
        paused_trials: List[Tuple[str, int]],
        min_resource: int,
        skip_optimization: bool,
    ) -> Dict[str, Any]:
        """
        See :meth:`DynamicHPOSearcher.score_paused_trials_and_new_configs`.
        If ``skip_optimization == True``, this is passed to the posterior state
        computation, and refitting of the surrogate model is skipped. Otherwise,
        nothing is passed, so the built in ``skip_optimization`` logic is used.
        """
        # Test whether we are still at the beginning, where we always return
        # new configs from ``points_to_evaluate`` or drawn at random
        config = self.get_config()
        if INTERNAL_KEY not in config:
            return {"config": config}
        # Collect all extended configs to be scored
        state = self.state_transformer.state
        assert (
            not state.hp_ranges.is_attribute_fixed()
        ), "Internal error: state.hp_ranges.is_attribute_fixed() must not be True"
        # Paused trials are scored at the next level they attain, while
        # ``paused_trials`` lists their current level
        configs_paused = [
            self.config_space_ext.get(state.config_for_trial[trial_id], resource + 1)
            for trial_id, resource in paused_trials
        ]
        resources_all = [x[1] + 1 for x in paused_trials]
        num_new = max(self.num_initial_candidates, len(paused_trials))
        exclusion_candidates = self._get_exclusion_candidates()
        # New configurations are scored at level ``min_resource``
        configs_new = [
            self.config_space_ext.get(config, min_resource)
            for config in self.random_generator.generate_candidates_en_bulk(
                num_new, exclusion_list=exclusion_candidates
            )
        ]
        num_new = len(configs_new)  # Can be less than before
        configs_all = configs_paused + configs_new
        resources_all.extend([min_resource] * num_new)
        num_all = len(resources_all)

        # Score all extended configurations :math:`(x, r)` with
        # :math:`EI(x | r)`, expected improvement at level :math:`r`. Note that
        # the incumbent is the best value at the same level, not overall. This
        # is why we compute the score values in chunks for the same :math:`r`
        # value
        if skip_optimization:
            kwargs = dict(skip_optimization=skip_optimization)
        else:
            kwargs = dict()
        # Note: Asking for the model triggers the posterior computation
        model = self.state_transformer.model(**kwargs)
        resources_all = np.array(resources_all)
        sorted_ind = np.argsort(resources_all)
        resources_all = resources_all[sorted_ind]
        # Find positions where resource value changes
        change_pos = (
            [0]
            + list(np.flatnonzero(resources_all[:-1] != resources_all[1:]) + 1)
            + [num_all]
        )
        scores = np.empty((num_all,))
        for start, end in zip(change_pos[:-1], change_pos[1:]):
            resource = resources_all[start]
            assert resources_all[end - 1] == resource

            def my_filter_observed_data(config: Configuration) -> bool:
                return self.config_space_ext.get_resource(config) == resource

            # If there is data at level ``resource``, the incumbent in EI
            # should only be computed over this. If there is no data at level
            # ``resource``, the incumbent is computed over all data
            if state.num_observed_cases(resource=resource) > 0:
                filter_observed_data = my_filter_observed_data
            else:
                filter_observed_data = None
            model.set_filter_observed_data(filter_observed_data)
            candidates_scorer = create_initial_candidates_scorer(
                initial_scoring="acq_func",
                model=model,
                acquisition_class=self.acquisition_class,
                random_state=self.random_state,
            )
            ind_for_resource = sorted_ind[start:end]
            scores_for_resource = candidates_scorer.score(
                [configs_all[pos] for pos in ind_for_resource]
            )
            scores[ind_for_resource] = scores_for_resource

        # Pick the winner
        best_ind = np.argmin(scores)
        if best_ind < len(paused_trials):
            return {"trial_id": paused_trials[best_ind][0]}
        else:
            return {"config": self._postprocess_config(configs_all[best_ind])}


class DynamicHPOSearcher(BaseSearcher):
    """
    Supports model-based decisions in the DyHPO algorithm proposed by Wistuba
    etal (see
    :class:`~syne_tune.optimizer.schedulers.searchers.dyhpo.DyHPORungSystem`).

    It is *not* recommended to create :class:`DynamicHPOSearcher` searcher
    objects directly, but rather to create
    :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler` objects with
    ``searcher="dyhpo"`` and ``type="dyhpo"``, and passing arguments here in
    ``search_options``. This will use the appropriate functions from
    :mod:``syne_tune.optimizer.schedulers.searchers.gp_searcher_factory`` to
    create components in a consistent way.

    This searcher is special, in that it contains a searcher of type
    :class:`~syne_tune.optimizer.schedulers.searchers.GPMultiFidelitySearcher`.
    Also, its model-based scoring is not triggered by :meth:`get_config`, but
    rather when the scheduler tries to find a trial which can be promoted. At
    this point, :meth:`score_paused_trials_and_new_configs` is called, which
    scores all paused trials along with new configurations. Depending on who
    is the best scorer, a paused trial is resumed, or a trial with a new
    configuration is started. Since all the work is already done in
    :meth:`score_paused_trials_and_new_configs`, the implementation of
    :meth:`get_config` becomes trivial. Extra points:

    * The number of new configurations scored in
      :meth:`score_paused_trials_and_new_configs` is the maximum of
      ``num_init_candidates`` and the number of paused trials scored as well
    * The parameters of the surrogate model are not refit in every call of
      :meth:`score_paused_trials_and_new_configs`, but only when in the last
      recent call, a new configuration was chosen as top scorer. The aim is
      to do refitting in a similar frequency to MOBSTER, where decisions on
      whether to resume a trial are not done in a model-based way.

    This searcher must be used with
    :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler` and
     ``type="dyhpo"``. It has the same constructor parameters as
    :class:`~syne_tune.optimizer.schedulers.searchers.GPMultiFidelitySearcher`.
    Of these, the following are not used, but need to be given valid values:
    ``resource_acq``, ``initial_scoring``, ``skip_local_optimization``, so that
    a ``GPMultiFidelitySearcher`` can be created.
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        metric: str,
        points_to_evaluate: Optional[List[dict]] = None,
        **kwargs,
    ):
        mode = kwargs.get("mode")
        if mode is None:
            mode = "min"
        super().__init__(
            config_space, metric, points_to_evaluate=points_to_evaluate, mode=mode
        )
        if "searcher_int" in kwargs:
            self._searcher_int = kwargs["searcher_int"]
        else:
            assert (
                kwargs.get("model") != "gp_independent"
            ), "model='gp_independent' is not supported"
            self._searcher_int = MyGPMultiFidelitySearcher(
                config_space,
                metric=metric,
                points_to_evaluate=points_to_evaluate,
                **kwargs,
            )
        self._previous_winner_new_trial = True

    def configure_scheduler(self, scheduler):
        from syne_tune.optimizer.schedulers import HyperbandScheduler

        self._searcher_int.configure_scheduler(scheduler)
        err_msg = (
            "This searcher requires HyperbandScheduler scheduler with type='dyhpo'"
        )
        assert isinstance(scheduler, HyperbandScheduler), err_msg
        assert scheduler.scheduler_type == "dyhpo", err_msg

    def get_config(self, **kwargs) -> Optional[dict]:
        assert (
            "new_config" in kwargs
        ), "Internal error: 'new_config' argument must be given"
        return kwargs["new_config"]

    def on_trial_result(
        self,
        trial_id: str,
        config: Dict[str, Any],
        result: Dict[str, Any],
        update: bool,
    ):
        self._searcher_int.on_trial_result(trial_id, config, result, update)

    def register_pending(
        self,
        trial_id: str,
        config: Optional[dict] = None,
        milestone: Optional[int] = None,
    ):
        self._searcher_int.register_pending(trial_id, config, milestone)

    def remove_case(self, trial_id: str, **kwargs):
        self._searcher_int.remove_case(trial_id, **kwargs)

    def evaluation_failed(self, trial_id: str):
        self._searcher_int.evaluation_failed(trial_id)

    def cleanup_pending(self, trial_id: str):
        self._searcher_int.cleanup_pending(trial_id)

    def dataset_size(self):
        return self._searcher_int.dataset_size()

    def model_parameters(self):
        return self._searcher_int.model_parameters()

    def score_paused_trials_and_new_configs(
        self,
        paused_trials: List[Tuple[str, int]],
        min_resource: int,
    ) -> Dict[str, Any]:
        """
        This method computes acquisition scores for a number of extended
        configs :math:`(x, r)`. The acquisition score :math:`EI(x | r + 1)` is
        expected improvement (EI) at resource level :math:`r + 1`. Here, the
        incumbent used in EI is the best value attained at level :math:`r + 1`,
        or the best value overall if there is no data yet at that level.
        There are two types of configs being scored:

        * Paused trials: Passed by ``paused_trials`` as tuples
          ``(trial_id, resource)``
        * New configurations drawn at random. For these, the score is EI
          at :math:`r` equal to ``min_resource``

        We return a dictionary. If a paused trial wins, its ``trial_id`` is
        returned with key "trial_id". If a new configuration wins, this
        configuration is returned with key "config".

        Note: As long as the internal searcher still returns configs from
        ``points_to_evaluate`` or drawn at random, this method always returns
        this config with key "config". Scoring and considering paused trials
        is only done afterwards.

        :param paused_trials: See above. Can be empty
        :param min_resource:
        :return: Dictionary, see above
        """
        result = self._searcher_int.score_paused_trials_and_new_configs(
            paused_trials=paused_trials,
            min_resource=min_resource,
            skip_optimization=not self._previous_winner_new_trial,
        )
        self._previous_winner_new_trial = "config" in result
        return result

    def get_state(self) -> Dict[str, Any]:
        return {
            "searcher_int": self._searcher_int.get_state(),
            "previous_winner_new_trial": self._previous_winner_new_trial,
        }

    def _restore_from_state(self, state: Dict[str, Any]):
        raise NotImplementedError

    def clone_from_state(self, state: Dict[str, Any]):
        searcher_int = self._searcher_int.clone_from_state(state["searcher_int"])
        return DynamicHPOSearcher(
            self.config_space,
            metric=self._metric,
            mode=self._mode,
            searcher_int=searcher_int,
        )

    @property
    def debug_log(self) -> Optional[DebugLogPrinter]:
        return self._searcher_int.debug_log
