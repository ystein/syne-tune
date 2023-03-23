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
from typing import List, Tuple
import numpy as np
import logging

from syne_tune.try_import import try_import_gpsearchers_message

try:
    from scipy.special import xlogy
except ImportError:
    logging.info(try_import_gpsearchers_message())

from syne_tune.tuner_callback import TunerCallback
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
        self._trials_with_checkpoints_removed = None
        self._prob_new_is_better = prob_new_is_better

    def _check_and_initialize(self, tuner):
        scheduler = tuner.scheduler
        assert isinstance(
            scheduler, HyperbandScheduler
        ), f"This callback only supports HyperbandScheduler"
        self._terminator = scheduler.terminator
        self._tuner = tuner  # Do we need this?

    def on_tuning_start(self, tuner):
        self._check_and_initialize(tuner)
        # Contains ``(trial_id, level)`` for paused trials whose checkpoints
        # were removed.
        self._trials_with_checkpoints_removed = set()

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
    ) -> List[Tuple[str, float]]:
        r"""
        Computes scores for paused trials in ``trials_to_score``, with entries
        ``(trial_id, rank, metric_val, level)``. These are approximations of
        expected cost of checkpoint removal. ``time_ratio`` is the ratio between
        time left and time already spent for the experiment, called :math:`\beta`
        in the note.

        :param trials_to_score: See above
        :param time_ratio: See above
        :return: List of ``(trial_id, score_val)``
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
        levels = np.array(levels)
        q_probs = ((time_ratio + 1) * prom_quants - ranks / lens) / time_ratio
        # TODO: Probability should depend on level
        p_probs = np.full_like(q_probs, self._prob_new_is_better)
        scores = np.log(levels) / time_ratio - lens * _bernoulli_relative_entropy(
            q_probs, p_probs
        )
        return list(zip(trial_ids, scores))

    def on_loop_end(self):
        for trial_id in self._tuner.scheduler.trials_checkpoints_can_be_removed():
            self._tuner.trial_backend.delete_checkpoint(trial_id)
