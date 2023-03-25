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
from typing import Optional

from syne_tune.tuner_callback import TunerCallback
from syne_tune.callbacks.hyperband_remove_checkpoints_callback import (
    HyperbandRemoveCheckpointsCallback,
)
from syne_tune.optimizer.scheduler import TrialScheduler
from syne_tune.optimizer.schedulers import HyperbandScheduler
from syne_tune.optimizer.schedulers.hyperband_promotion import PromotionRungSystem


def early_checkpoint_removal_factory(
    scheduler: TrialScheduler,
) -> Optional[TunerCallback]:
    """
    Early checkpoint removal is implemented by callbacks, which depend on which
    scheduler is being used. For many schedulers, early checkpoint removal is
    not supported.

    :param scheduler: Scheduler for which early checkpoint removal is requested
    :param kwargs: Arguments to constructor of callback
    :return: Callback for early checkpoint removal, or ``None`` if this is not
        supported for the scheduler
    """
    callback = None
    if isinstance(scheduler, HyperbandScheduler) and isinstance(
        scheduler.terminator, PromotionRungSystem
    ):
        callback = HyperbandRemoveCheckpointsCallback(
            **scheduler.params_early_checkpoint_removal()
        )
    return callback
