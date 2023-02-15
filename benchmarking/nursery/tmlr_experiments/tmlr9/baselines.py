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
from benchmarking.commons.default_baselines import (
    BayesianOptimization,
    MOBSTER,
    HyperTune,
    DyHPO,
)


class Methods:
    BO_SKIP10 = "BO-SKIP10"
    MOBSTER_SKIP10 = "MOBSTER-SKIP10"
    HYPERTUNE_SKIP10 = "HyperTune-SKIP10"
    DYHPO_SKIP10 = "DYHPO-SKIP10"


methods = {
    Methods.BO_SKIP10: lambda method_arguments: BayesianOptimization(
        method_arguments,
        search_options=dict(
            opt_skip_init_length=400,
            opt_skip_period=10,
        ),
    ),
    Methods.MOBSTER_SKIP10: lambda method_arguments: MOBSTER(
        method_arguments,
        type="promotion",
        search_options=dict(
            opt_skip_init_length=400,
            opt_skip_period=10,
        ),
    ),
    Methods.HYPERTUNE_SKIP10: lambda method_arguments: HyperTune(
        method_arguments,
        type="promotion",
        search_options=dict(
            opt_skip_init_length=400,
            opt_skip_period=10,
        ),
    ),
    Methods.DYHPO_SKIP10: lambda method_arguments: DyHPO(
        method_arguments,
        search_options=dict(
            opt_skip_init_length=400,
            opt_skip_period=10,
        ),
    ),
}
