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
"""
Example for speculative checkpoint removal with asynchronous multi-fidelity
"""
import logging

from syne_tune.backend import LocalBackend
from syne_tune.callbacks.hyperband_remove_checkpoints_callback import (
    HyperbandRemoveCheckpointsCallback,
)
from syne_tune.optimizer.baselines import MOBSTER
from syne_tune import Tuner, StoppingCriterion

from benchmarking.commons.benchmark_definitions.mlp_on_fashionmnist import (
    mlp_fashionmnist_benchmark,
)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    random_seed = 31415927
    n_workers = 4
    max_num_checkpoints = 10
    # This time may be too short to see positive effects:
    max_wallclock_time = 1800

    # We pick the MLP on FashionMNIST benchmark
    benchmark = mlp_fashionmnist_benchmark()

    # Local backend
    # By setting ``delete_checkpoints=True``, we ask for checkpoints to be removed
    # once a trial cannot be resumed anymore
    trial_backend = LocalBackend(
        entry_point=str(benchmark.script),
        delete_checkpoints=True,
    )

    # MOBSTER (model-based ASHA) with promotion scheduling (pause and resume).
    # Checkpoints are written for each paused trial, and these are not removed,
    # because in principle, every paused trial may be resumed in the future.
    # If checkpoints are large, this may fill up your disk.
    # Here, we use speculative checkpoint removal to keep the number of checkpoints
    # to at most ``max_num_checkpoints``. To this end, paused trials are ranked by
    # expected cost of removing their checkpoint.
    scheduler = MOBSTER(
        benchmark.config_space,
        type="promotion",
        max_resource_attr=benchmark.max_resource_attr,
        resource_attr=benchmark.resource_attr,
        mode=benchmark.mode,
        metric=benchmark.metric,
        random_seed=random_seed,
        early_checkpoint_removal_kwargs=dict(
            max_num_checkpoints=max_num_checkpoints,
        ),
    )

    stop_criterion = StoppingCriterion(max_wallclock_time=max_wallclock_time)
    # The tuner activates (early) checkpoint iff ``trial_backend.delete_checkpoints``.
    # In this case, it requests details from the scheduler (which is
    # ``early_checkpoint_removal_kwargs`` in our case).
    # Early checkpoint removal is done by appending a callback to those normally used
    # with the tuner.
    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=n_workers,
    )
    tuner.run()

    # We can obtain details about checkpoint removal from the callback, which is the
    # last one in ``tuner``
    callback = tuner.callbacks[-1]
    assert isinstance(callback, HyperbandRemoveCheckpointsCallback), (
        "The final callback in tuner.callbacks should be "
        f"HyperbandRemoveCheckpointsCallback, but is {type(callback)}"
    )
    logging.info(f"Number of checkpoints removed: {callback.num_checkpoints_removed}")
    trials_resumed = callback.trials_resumed_without_checkpoint()
    if trials_resumed:
        logging.info(
            f"The following {len(trials_resumed)} trials were resumed without a checkpoint:\n{trials_resumed}"
        )
        sum_resource = sum(level for _, level in trials_resumed)
    else:
        logging.info("No trials were resumed without a checkpoint")
        sum_resource = 0
    logging.info(f"Cost: {sum_resource} epochs of training from scratch")
