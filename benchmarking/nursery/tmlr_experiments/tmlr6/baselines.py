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
    SyncHyperband,
    SyncMOBSTER,
    SyncBOHB,
    DEHB,
)
from syne_tune.optimizer.schedulers import HyperbandScheduler


class Methods:
    ASHA = "ASHA"
    MOBSTER_JOINT = "MOBSTER-JOINT"
    MOBSTER_INDEP = "MOBSTER-INDEP"
    MOBSTER_M52_RW = "MOBSTER-M52-RW"
    HYPERTUNE_INDEP = "HyperTune-INDEP"
    HYPERTUNE_JOINT = "HyperTune-JOINT"
    BOHB = "BOHB"
    BORE = "BORE"
    SYNCHB = "SyncHB"
    SYNCMOBSTER = "SyncMOBSTER"
    SYNCBOHB = "SyncBOHB"
    DEHB = "DEHB"


methods = {
    Methods.ASHA: lambda method_arguments: ASHA(
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
    Methods.MOBSTER_JOINT: lambda method_arguments: MOBSTER(
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
    Methods.MOBSTER_INDEP: lambda method_arguments: MOBSTER(
        config_space=method_arguments.config_space,
        search_options=dict(
            search_options(method_arguments),
            model="gp_independent",
        ),
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
    Methods.MOBSTER_M52_RW: lambda method_arguments: MOBSTER(
        config_space=method_arguments.config_space,
        search_options=dict(
            search_options(method_arguments),
            gp_resource_kernel="matern52-res-warp",
        ),
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
    Methods.HYPERTUNE_INDEP: lambda method_arguments: HyperTune(
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
    Methods.HYPERTUNE_JOINT: lambda method_arguments: HyperTune(
        config_space=method_arguments.config_space,
        search_options=dict(
            search_options(method_arguments),
            model="gp_multitask",
        ),
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
    Methods.BOHB: lambda method_arguments: HyperbandScheduler(
        config_space=method_arguments.config_space,
        searcher="kde",
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
    Methods.BORE: lambda method_arguments: HyperbandScheduler(
        config_space=method_arguments.config_space,
        searcher="bore",
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
    Methods.SYNCHB: lambda method_arguments: SyncHyperband(
        config_space=method_arguments.config_space,
        search_options=search_options(method_arguments),
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_resource_attr=method_arguments.max_resource_attr,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
    ),
    Methods.SYNCMOBSTER: lambda method_arguments: SyncMOBSTER(
        config_space=method_arguments.config_space,
        search_options=search_options(method_arguments),
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_resource_attr=method_arguments.max_resource_attr,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
    ),
    Methods.SYNCBOHB: lambda method_arguments: SyncBOHB(
        config_space=method_arguments.config_space,
        search_options=search_options(method_arguments),
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_resource_attr=method_arguments.max_resource_attr,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
    ),
    Methods.DEHB: lambda method_arguments: DEHB(
        config_space=method_arguments.config_space,
        search_options=search_options(method_arguments),
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_resource_attr=method_arguments.max_resource_attr,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
    ),
}
