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
from typing import List, Tuple, Dict, Any
import numpy as np
import logging
from operator import itemgetter
import time

from syne_tune.try_import import try_import_gpsearchers_message

try:
    from scipy.special import xlogy
except ImportError:
    logging.info(try_import_gpsearchers_message())

from syne_tune.tuner_callback import TunerCallback
from syne_tune.backend.simulator_backend.simulator_backend import SimulatorBackend
from syne_tune.backend.trial_status import Trial
from syne_tune.optimizer.scheduler import SchedulerDecision
from syne_tune.optimizer.schedulers import HyperbandScheduler
from syne_tune.optimizer.schedulers.hyperband_stopping import PausedTrialsResult


def _q_log_q_div_p(q_probs: np.ndarray, p_probs: np.ndarray) -> np.ndarray:
    return xlogy(q_probs, q_probs) - xlogy(q_probs, p_probs)


def _bernoulli_relative_entropy(q_probs: np.ndarray, p_probs: np.ndarray) -> np.ndarray:
    q_clipped = np.clip(q_probs, 1e-7, 1 - 1e-7)
    result = _q_log_q_div_p(q_clipped, p_probs) + _q_log_q_div_p(
        1.0 - q_clipped, 1.0 - p_probs
    )
    result[q_probs >= 1] = 0
    result[q_probs < 0] = np.inf
    return result


class TrialStatus:
    RUNNING = 0
    PAUSED_WITH_CHECKPOINT = 1
    PAUSED_NO_CHECKPOINT = 2
    STOPPED_OR_COMPLETED = 3


class HyperbandRemoveCheckpointsCallback(TunerCallback):
    """
    Implements speculative early removal of checkpoints of paused trials for
    :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler` (only for
    types which pause trials at rung levels).

    In this scheduler, any paused trial can in principle be resumed in the
    past, which is why we remove checkpoints speculatively. The idea is to
    keep the total number of checkpoints no larger than ``max_num_checkpoints``.
    If this limit is reached, we rank all currently paused trials which still
    have a checkpoint and remove checkpoints for those who score worst. If a
    trial is resumed whose checkpoint has been removed, we have to train from
    scratch, at a cost proportional to the rung level the trial is paused at.
    The score is an approximation to this expected cost, the product of rung
    level and probability of getting resumed. This probability depends on the
    current rung size, the rank of the trial in the rung, and both the time
    spent and remaining for the experiment, so we need ``max_wallclock_time``.
    Details are given in a technical report.

    ``prob_new_is_better`` is the probability that a new trial arriving at a
    rung ranks better than an existing paused one.

    TODO: This probability should be estimated, even for every rung level!

    .. note::
       The true maximum number of trials with checkpoints may be larger than
       ``max_num_checkpoints``. The number is determined here by
       :meth:`on_trial_result` and :meth:`on_trial_complete`, so that trials
       are only counted once they report for the first time. Also, if a paused
       trial without checkpoint is resumed, it is counted only with its first
       report.

    :param max_num_checkpoints: Once the total number of checkpoints surpasses
        this number, we remove some.
    :param max_wallclock_time: Maximum time of the experiment
    :param prob_new_is_better: See above
    """

    def __init__(
        self,
        max_num_checkpoints: int,
        max_wallclock_time: int,
        prob_new_is_better: float,
    ):
        self._tuner = None
        self.max_num_checkpoints = max_num_checkpoints
        self._max_wallclock_time = max_wallclock_time
        self._prob_new_is_better = prob_new_is_better
        self._trials_with_checkpoints_removed = None
        self._trial_status = None
        self._start_time = None

    def _check_and_initialize(self, tuner):
        scheduler = tuner.scheduler
        assert isinstance(
            scheduler, HyperbandScheduler
        ), f"This callback only supports HyperbandScheduler"
        self._terminator = scheduler.terminator
        self._trial_backend = tuner.trial_backend
        assert not isinstance(
            self._trial_backend, SimulatorBackend
        ), "Checkpoint removal not supported for SimulatorBackend"

    def on_tuning_start(self, tuner):
        self._check_and_initialize(tuner)
        # Contains ``(trial_id, level)`` for paused trials whose checkpoints
        # were removed. Note that ``trial_id`` is not enough, since a paused trial
        # can have its checkpoint removed, then be resumed and paused again (at a
        # higher rung level) with a checkpoint
        self._trials_with_checkpoints_removed = set()
        # Maps ``trial_id`` to ``TrialStatus``. Used to count the number of
        # trials with checkpoints
        self._trial_status = dict()
        self._start_time = time.perf_counter()

    def _filter_paused_trials(
        self, paused_trials: PausedTrialsResult
    ) -> PausedTrialsResult:
        return [
            entry
            for entry in paused_trials
            if (entry[0], entry[-1]) not in self._trials_with_checkpoints_removed
        ]

    def _compute_scores(
        self, trials_to_score: List[Tuple[str, int, float, int]], time_ratio: float
    ) -> List[Tuple[str, float, int]]:
        r"""
        Computes scores for paused trials in ``trials_to_score``, with entries
        ``(trial_id, rank, metric_val, level)``. These are approximations of
        expected cost of checkpoint removal. ``time_ratio`` is the ratio between
        time left and time already spent for the experiment, called :math:`\beta`
        in the note.

        :param trials_to_score: See above
        :param time_ratio: See above
        :return: List of ``(trial_id, score_val, level)``
        """
        info_rungs = self._terminator.information_for_rungs()
        lens_rung = {x[0]: x[1] for x in info_rungs}
        prom_quants_rung = {x[0]: x[2] for x in info_rungs}
        trial_ids, ranks, lens, prom_quants, levels = zip(
            *[
                (trial_id, rank, lens_rung[level], prom_quants_rung[level], level)
                for trial_id, rank, _, level in trials_to_score
            ]
        )
        ranks = np.array(ranks)
        lens = np.array(lens)
        prom_quants = np.array(prom_quants)
        q_probs = ((time_ratio + 1) * prom_quants - ranks / lens) / time_ratio
        # TODO: Probability should depend on level
        p_probs = np.full_like(q_probs, self._prob_new_is_better)
        scores = np.log(levels) / time_ratio - lens * _bernoulli_relative_entropy(
            q_probs, p_probs
        )
        return list(zip(trial_ids, scores, levels))

    def _count_trials_with_checkpoints(self) -> int:
        has_checkpoint = {TrialStatus.RUNNING, TrialStatus.PAUSED_WITH_CHECKPOINT}
        return sum(status in has_checkpoint for status in self._trial_status.values())

    def _remove_checkpoint_of(self, trial_id: str, level: int):
        self._tuner.trial_backend.delete_checkpoint(int(trial_id))
        self._trial_status[trial_id] = TrialStatus.PAUSED_NO_CHECKPOINT
        self._trials_with_checkpoints_removed.add((trial_id, int(level)))

    def _get_time_ratio(self) -> float:
        current_time = time.perf_counter()
        time_elapsed = current_time - self._start_time
        time_left = self._max_wallclock_time - time_elapsed
        return max(time_left, 1e-3) / max(time_elapsed, 1e-3)

    def on_loop_end(self):
        num_trials_with_checkpoints = self._count_trials_with_checkpoints()
        num_to_remove = num_trials_with_checkpoints - self.max_num_checkpoints
        if num_to_remove > 0:
            paused_trials_with_checkpoints = self._filter_paused_trials(
                self._terminator.paused_trials()
            )
            time_ratio = self._get_time_ratio()
            scores = self._compute_scores(paused_trials_with_checkpoints, time_ratio)
            num = min(num_to_remove, len(paused_trials_with_checkpoints))
            for trial_id, _, level in sorted(scores, key=itemgetter(1))[:num]:
                self._remove_checkpoint_of(trial_id, level)

    def on_trial_complete(self, trial: Trial, result: Dict[str, Any]):
        trial_id = str(trial.trial_id)
        self._trial_status[trial_id] = TrialStatus.STOPPED_OR_COMPLETED

    def on_trial_result(
        self, trial: Trial, status: str, result: Dict[str, Any], decision: str
    ):
        trial_id = str(trial.trial_id)
        if decision == SchedulerDecision.CONTINUE:
            new_status = TrialStatus.RUNNING
        elif decision == SchedulerDecision.PAUSE:
            new_status = TrialStatus.PAUSED_WITH_CHECKPOINT
        else:
            assert decision == SchedulerDecision.STOP  # Sanity check
            new_status = TrialStatus.STOPPED_OR_COMPLETED
        self._trial_status[trial_id] = new_status
