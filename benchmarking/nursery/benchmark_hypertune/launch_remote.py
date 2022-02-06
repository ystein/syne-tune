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
from pathlib import Path

from benchmarking.commons.launch_remote import launch_remote
from benchmarking.commons.benchmark_definitions import benchmark_definitions


if __name__ == "__main__":
    from benchmarking.nursery.benchmark_hypertune.baselines import methods
    from benchmarking.nursery.benchmark_hypertune.benchmark_main import (
        extra_args,
        map_extra_args,
    )

    def _is_expensive_method(method: str) -> bool:
        return method.startswith("MOBSTER") or method.startswith("HYPERTUNE")

    entry_point = Path(__file__).parent / "benchmark_main.py"
    launch_remote(
        entry_point=entry_point,
        methods=methods,
        benchmark_definitions=benchmark_definitions,
        extra_args=extra_args,
        map_extra_args=map_extra_args,
        is_expensive_method=_is_expensive_method,
    )
