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
from typing import Dict, Any
from benchmarking.commons.baselines import (
    search_options,
)
from syne_tune.optimizer.baselines import (
    RandomSearch,
    BayesianOptimization,
    DyHPO,
    ASHA,
    MOBSTER,
)


class Methods:
    RS = "RS"
    BO = "BO"
    DYHPO = "DYHPO"
    ASHA = "ASHA"
    MOBSTER = "MOBSTER"


methods = {
    Methods.RS: lambda method_arguments: RandomSearch(
        config_space=method_arguments.config_space,
        search_options=search_options(method_arguments),
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_resource_attr=method_arguments.max_resource_attr,
        random_seed=method_arguments.random_seed,
    ),
    Methods.BO: lambda method_arguments: BayesianOptimization(
        config_space=method_arguments.config_space,
        search_options=search_options(method_arguments),
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_resource_attr=method_arguments.max_resource_attr,
        random_seed=method_arguments.random_seed,
    ),
    Methods.ASHA: lambda method_arguments: ASHA(
        config_space=method_arguments.config_space,
        type="promotion",
        search_options=search_options(method_arguments),
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_resource_attr=method_arguments.max_resource_attr,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
        brackets=method_arguments.num_brackets,
    ),
    Methods.DYHPO: lambda method_arguments: DyHPO(
        config_space=method_arguments.config_space,
        search_options=search_options(method_arguments),
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_resource_attr=method_arguments.max_resource_attr,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
        brackets=method_arguments.num_brackets,
    ),
    Methods.MOBSTER: lambda method_arguments: MOBSTER(
        config_space=method_arguments.config_space,
        type="promotion",
        search_options=search_options(method_arguments),
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_resource_attr=method_arguments.max_resource_attr,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
        brackets=method_arguments.num_brackets,
    ),
}
