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
from benchmarking.commons.baselines import (
    search_options,
)
from syne_tune.optimizer.baselines import (
    ASHA,
    MOBSTER,
    HyperTune,
)


class Methods:
    ASHA_STOP = "ASHA-STOP"
    MOBSTER_STOP = "MOBSTER-STOP"
    HYPERTUNE_STOP = "HyperTune-STOP"
    ASHA_PROM = "ASHA-PROM"
    MOBSTER_PROM = "MOBSTER-PROM"
    HYPERTUNE_PROM = "HyperTune-PROM"


methods = {
    Methods.ASHA_STOP: lambda method_arguments: ASHA(
        config_space=method_arguments.config_space,
        search_options=search_options(method_arguments),
        type="stopping",
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_resource_attr=method_arguments.max_resource_attr,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
        **(
            method_arguments.scheduler_kwargs
            if method_arguments.scheduler_kwargs is not None
            else dict()
        ),
    ),
    Methods.MOBSTER_STOP: lambda method_arguments: MOBSTER(
        config_space=method_arguments.config_space,
        search_options=search_options(method_arguments),
        type="stopping",
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_resource_attr=method_arguments.max_resource_attr,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
        **(
            method_arguments.scheduler_kwargs
            if method_arguments.scheduler_kwargs is not None
            else dict()
        ),
    ),
    Methods.HYPERTUNE_STOP: lambda method_arguments: HyperTune(
        config_space=method_arguments.config_space,
        search_options=search_options(method_arguments),
        type="stopping",
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_resource_attr=method_arguments.max_resource_attr,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
        **(
            method_arguments.scheduler_kwargs
            if method_arguments.scheduler_kwargs is not None
            else dict()
        ),
    ),
    Methods.ASHA_PROM: lambda method_arguments: ASHA(
        config_space=method_arguments.config_space,
        search_options=search_options(method_arguments),
        type="promotion",
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_resource_attr=method_arguments.max_resource_attr,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
        **(
            method_arguments.scheduler_kwargs
            if method_arguments.scheduler_kwargs is not None
            else dict()
        ),
    ),
    Methods.MOBSTER_PROM: lambda method_arguments: MOBSTER(
        config_space=method_arguments.config_space,
        search_options=search_options(method_arguments),
        type="promotion",
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_resource_attr=method_arguments.max_resource_attr,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
        **(
            method_arguments.scheduler_kwargs
            if method_arguments.scheduler_kwargs is not None
            else dict()
        ),
    ),
    Methods.HYPERTUNE_PROM: lambda method_arguments: HyperTune(
        config_space=method_arguments.config_space,
        search_options=search_options(method_arguments),
        type="promotion",
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_resource_attr=method_arguments.max_resource_attr,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
        **(
            method_arguments.scheduler_kwargs
            if method_arguments.scheduler_kwargs is not None
            else dict()
        ),
    ),
}
