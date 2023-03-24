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
    from scipy.special import betainc
except ImportError:
    logging.info(try_import_gpsearchers_message())

from syne_tune.tuner_callback import TunerCallback
from syne_tune.backend.simulator_backend.simulator_backend import SimulatorBackend
from syne_tune.backend.trial_status import Trial
from syne_tune.optimizer.scheduler import SchedulerDecision
from syne_tune.optimizer.schedulers import HyperbandScheduler
from syne_tune.optimizer.schedulers.hyperband_stopping import PausedTrialsResult


def _binomial_cdf(
    u_vals: np.ndarray, n_vals: np.ndarray, p_vals: np.ndarray
) -> np.ndarray:
    r"""
    Computes binomial cumulative distribution function :math:`P(X \le u)`, where
    :math:`X\sim \mathrm{bin}(n, p)`.

    :param u_vals: Values for :math:`u`
    :param n_vals: Values for :math:`n`
    :param p_vals: Values for :math:`p`
    :return: CDF values
    """
    a_vals = np.maximum(n_vals - u_vals, 1e-7)
    b_vals = np.maximum(u_vals + 1, 1e-7)
    result = betainc(a_vals, b_vals, 1 - p_vals)
    result[u_vals < 0] = 0
    result[u_vals >= n_vals] = 1
    return result


class TrialStatus:
    RUNNING = 0
    PAUSED_WITH_CHECKPOINT = 1
    PAUSED_NO_CHECKPOINT = 2
    STOPPED_OR_COMPLETED = 3


class BetaBinomialEstimator:
    """
    Estimator of the probability :math:`p = P(X = 1)` for a variable :math:`X`
    with Bernoulli distribution. This is using a Beta prior, which is
    conjugate to the binomial likelihood. The prior is parameterized by
    effective sample size ``beta_size`` (:math:`a + b`) and mean ``beta_mean``
    (:math:`a / (a + b)`).
    """

    def __init__(self, beta_mean: float, beta_size: float):
        assert 0 < beta_mean < 1
        assert beta_size > 0
        self._beta_a = beta_mean * beta_size
        self._beta_size = beta_size
        self._num_one = 0
        self._num_total = 0

    def update(self, data: List[bool]):
        self._num_one += sum(data)
        self._num_total += len(data)

    @property
    def num_one(self) -> int:
        return self._num_one

    @property
    def num_total(self):
        return self._num_total

    def posterior_mean(self) -> float:
        a_posterior = self._beta_a + self._num_one
        aplusb_posterior = self._beta_size + self._num_total
        return a_posterior / aplusb_posterior


class HyperbandRemoveCheckpointsCallback(TunerCallback):
    """
    Implements speculative early removal of checkpoints of paused trials for
    :class:`~syne_tune.optimizer.schedulers.HyperbandScheduler` (only for
    types which pause trials at rung levels).

    In this scheduler, any paused trial can in principle be resumed in the
    future, which is why we remove checkpoints speculatively. The idea is to
    keep the total number of checkpoints no larger than ``max_num_checkpoints``.
    If this limit is reached, we rank all currently paused trials which still
    have a checkpoint and remove checkpoints for those with lowest scores. If a
    trial is resumed whose checkpoint has been removed, we have to train from
    scratch, at a cost proportional to the rung level the trial is paused at.
    The score is an approximation to this expected cost, the product of rung
    level and probability of getting resumed. This probability depends on the
    current rung size, the rank of the trial in the rung, and both the time
    spent and remaining for the experiment, so we need ``max_wallclock_time``.
    Details are given in a technical report.

    The probability of getting resumed also depends on the probability
    :math:`p_r` that a new trial arriving at rung :math:`r` ranks better than
    an existing paused one with a checkpoint. These probabilities are estimated
    here. For each new arrival at a rung, we obtain one datapoint for every
    paused trial with checkpoint there. We use Bayesian estimators with Beta
    prior given by mean ``prior_beta_mean`` and sample size ``prior_beta_size``.
    The mean should be :math:`< 1/2`). We also run an estimator for an overall
    probability :math:`p`, which is fed by all datapoints. This estimator is
    used as long as there are less than :math:`min_data_at_rung` datapoints at
    rung :math:`r`.

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
    :param metric: Name of metric in ``result`` of :meth:`on_trial_result`
    :param resource_attr: Name of resource attribute in ``result`` of
        :meth:`on_trial_result`
    :param mode: "min" or "max"
    :param prior_beta_mean: Parameter of Beta prior for estimators
    :param prior_beta_size: Parameter of Beta prior for estimators
    :param min_data_at_rung: See above
    """

    def __init__(
        self,
        max_num_checkpoints: int,
        max_wallclock_time: int,
        metric: str,
        resource_attr: str,
        mode: str,
        prior_beta_mean: float,
        prior_beta_size: float,
        min_data_at_rung: int = 5,
    ):
        self.max_num_checkpoints = max_num_checkpoints
        self._max_wallclock_time = max_wallclock_time
        self._metric = metric
        self._resource_attr = resource_attr
        assert mode in ["min", "max"]
        self._metric_sign = -1 if mode == "max" else 1
        self._prior_beta_mean = prior_beta_mean
        self._prior_beta_size = prior_beta_size
        self._min_data_at_rung = min_data_at_rung
        self._trials_with_checkpoints_removed = None
        self._trial_status = None
        self._estimator_for_rung = None
        self._estimator_overall = None
        self._start_time = None
        self._terminator = None
        self._trial_backend = None

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
        # See header comment
        self._estimator_for_rung = dict()
        self._estimator_overall = BetaBinomialEstimator(
            beta_mean=self._prior_beta_mean, beta_size=self._prior_beta_size
        )

    def _filter_paused_trials(
        self, paused_trials: PausedTrialsResult
    ) -> PausedTrialsResult:
        return [
            entry
            for entry in paused_trials
            if (entry[0], entry[-1]) not in self._trials_with_checkpoints_removed
        ]

    def estimator_for_rung(self, level: int) -> BetaBinomialEstimator:
        if level not in self._estimator_for_rung:
            self._estimator_for_rung[level] = BetaBinomialEstimator(
                beta_mean=self._prior_beta_mean, beta_size=self._prior_beta_size
            )
        return self._estimator_for_rung[level]

    def _probability_for_rung(self, level: int) -> float:
        estimator = self.estimator_for_rung(level)
        if estimator.num_total < self._min_data_at_rung:
            estimator = self._estimator_overall
        return estimator.posterior_mean()

    def _prepare_score_inputs(
        self, trials_to_score: List[Tuple[str, int, float, int]]
    ) -> (List[str], List[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        info_rungs = self._terminator.information_for_rungs()
        lens_rung = {r: n for r, n, _ in info_rungs}
        prom_quants_rung = {r: alpha for r, _, alpha in info_rungs}
        pvals_rung = {r: self._probability_for_rung(r) for r, _, _ in info_rungs}
        trial_ids, ranks, levels, rung_lens, prom_quants, p_vals = zip(
            *[
                (
                    trial_id,
                    rank + 1,
                    level,
                    lens_rung[level],
                    prom_quants_rung[level],
                    pvals_rung[level],
                )
                for trial_id, rank, _, level in trials_to_score
            ]
        )
        ranks = np.array(ranks)  # ranks starting from 1 (not 0)
        rung_lens = np.array(rung_lens)
        prom_quants = np.array(prom_quants)
        p_vals = np.array(p_vals)
        return trial_ids, levels, ranks, rung_lens, prom_quants, p_vals

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
        (
            trial_ids,
            levels,
            ranks,
            rung_lens,
            prom_quants,
            p_vals,
        ) = self._prepare_score_inputs(trials_to_score)
        n_vals = time_ratio * rung_lens
        u_vals = (time_ratio + 1) * prom_quants * rung_lens - ranks
        scores = _binomial_cdf(u_vals=u_vals, n_vals=n_vals, p_vals=p_vals) * np.array(
            levels
        )
        return list(zip(trial_ids, scores, levels))

    def _count_trials_with_checkpoints(self) -> int:
        has_checkpoint = {TrialStatus.RUNNING, TrialStatus.PAUSED_WITH_CHECKPOINT}
        return sum(status in has_checkpoint for status in self._trial_status.values())

    def _remove_checkpoint_of(self, trial_id: str, level: int):
        self._trial_backend.delete_checkpoint(int(trial_id))
        self._trial_status[trial_id] = TrialStatus.PAUSED_NO_CHECKPOINT
        self._trials_with_checkpoints_removed.add((trial_id, int(level)))

    def _get_time_ratio(self) -> float:
        current_time = time.perf_counter()
        time_elapsed = current_time - self._start_time
        time_remaining = self._max_wallclock_time - time_elapsed
        return max(time_remaining, 1e-3) / max(time_elapsed, 1e-3)

    def on_loop_end(self):
        num_to_remove = self._count_trials_with_checkpoints() - self.max_num_checkpoints
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

    def _update_estimators(self, trial_id: str, result: Dict[str, Any]):
        metric_val = float(result[self._metric])
        level = int(result[self._resource_attr])
        # Paused trials with checkpoints at rung level ``level``
        paused_trials_with_checkpoints = self._filter_paused_trials(
            self._terminator.paused_trials(resource=level)
        )
        # How many of these are better/worse than the new arrival? Note that the
        # new arrival may already be in that list
        data = [
            self._metric_sign * (metric_val - mv) <= 0
            for tid, _, mv, _ in paused_trials_with_checkpoints
            if tid != trial_id
        ]
        if data:
            for estimator in [self.estimator_for_rung(level), self._estimator_overall]:
                estimator.update(data)

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
        # This new arrival provides data for updating our estimators
        self._update_estimators(trial_id, result)
