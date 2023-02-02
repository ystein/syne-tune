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

from syne_tune.optimizer.schedulers.searchers import (
    BaseSearcher,
    GPMultiFidelitySearcher,
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
        self, paused_trials: List[Tuple[str, int]]
    ) -> Dict[str, Any]:
        """
        See :meth:`DynamicHPOSearcher.score_paused_trials_and_new_configs`.
        """
        # Test whether we are still at the beginning, where we always return
        # new configs from ``points_to_evaluate`` or drawn at random
        config = self.get_config()
        if INTERNAL_KEY not in config:
            return {"config": config}
        # Collect all external configs to be scored
        # - Use random_generator to select random configs
        # - Use initial_candidates_scorer to score
        # BUT: Marginals and EI scores need to be computed for extended configs,
        # not fixing the resource


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
        self._previous_winner_new_trial = False

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
        self, paused_trials: List[Tuple[str, int]]
    ) -> Dict[str, Any]:
        """
        This method computes acquisition scores for a number of extended
        configs :math:`(x, r)`. The acquisition score is expected improvement
        (EI) at resource level :math:`r + 1`. There are two types of configs
        being scored:

        * Paused trials: Passed by ``paused_trials`` as tuples
          ``(trial_id, resource)``
        * New configurations drawn at random. For these, :math:`r = 0`, so the
          score is EI at :math:`r = 1`

        We return a dictionary. If a paused trial wins, its ``trial_id`` is
        returned with key "trial_id". If a new configuration wins, this
        configuration is returned with key "config".

        Note: As long as the internal searcher still returns configs from
        ``points_to_evaluate`` or drawn at random, this method always returns
        this config with key "config". Scoring and considering paused trials
        is only done afterwards.

        :param paused_trials: See above. Can be empty
        :return: Dictionary, see above
        """
        return self._searcher_int.score_paused_trials_and_new_configs(paused_trials)

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
