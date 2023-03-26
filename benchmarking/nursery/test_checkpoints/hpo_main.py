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
import logging

from benchmarking.commons.hpo_main_local import main
from benchmarking.nursery.test_checkpoints.baselines import methods
from benchmarking.commons.benchmark_definitions import (
    real_benchmark_definitions as benchmark_definitions,
)
from syne_tune import Tuner
from syne_tune.callbacks.hyperband_remove_checkpoints_callback import (
    HyperbandRemoveCheckpointsCallback,
)


def post_processing(tuner: Tuner):
    callback = tuner.callbacks[-1]
    if isinstance(callback, HyperbandRemoveCheckpointsCallback):
        trials_resumed = callback.trials_resumed_without_checkpoint()
        if trials_resumed:
            logging.info(
                f"The following trials were resumed without a checkpoint:\n{trials_resumed}"
            )
        else:
            logging.info("No trials were resumed without a checkpoint")
    else:
        logging.info(
            "Final callback of Tuner is not HyperbandRemoveCheckpointsCallback"
        )


if __name__ == "__main__":
    main(methods, benchmark_definitions, post_processing=post_processing)
