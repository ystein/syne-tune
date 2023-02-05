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
    default_arguments,
    MethodArguments,
)
from syne_tune.optimizer.baselines import (
    RandomSearch as _RandomSearch,
    BayesianOptimization as _BayesianOptimization,
    DyHPO as _DyHPO,
    ASHA as _ASHA,
    MOBSTER as _MOBSTER,
    HyperTune as _HyperTune,
    BOHB as _BOHB,
    SyncHyperband as _SyncHyperband,
    SyncBOHB as _SyncBOHB,
    DEHB as _DEHB,
    SyncMOBSTER as _SyncMOBSTER,
    KDE as _KDE,
)
from syne_tune.util import recursive_merge


def RandomSearch(method_arguments: MethodArguments, **kwargs):
    return _RandomSearch(
        **recursive_merge(
            default_arguments(
                method_arguments,
                dict(
                    config_space=method_arguments.config_space,
                    search_options=search_options(method_arguments),
                ),
            ),
            kwargs,
        )
    )


def BayesianOptimization(method_arguments: MethodArguments, **kwargs):
    return _BayesianOptimization(
        **recursive_merge(
            default_arguments(
                method_arguments,
                dict(
                    config_space=method_arguments.config_space,
                    search_options=search_options(method_arguments),
                ),
            ),
            kwargs,
        )
    )


def ASHA(method_arguments: MethodArguments, **kwargs):
    return _ASHA(
        **recursive_merge(
            default_arguments(
                method_arguments,
                dict(
                    config_space=method_arguments.config_space,
                    search_options=search_options(method_arguments),
                    resource_attr=method_arguments.resource_attr,
                ),
            ),
            kwargs,
        )
    )


def DyHPO(method_arguments: MethodArguments, **kwargs):
    return _DyHPO(
        **recursive_merge(
            default_arguments(
                method_arguments,
                dict(
                    config_space=method_arguments.config_space,
                    search_options=search_options(method_arguments),
                    resource_attr=method_arguments.resource_attr,
                ),
            ),
            kwargs,
        )
    )


def HyperTune(method_arguments: MethodArguments, **kwargs):
    return _HyperTune(
        **recursive_merge(
            default_arguments(
                method_arguments,
                dict(
                    config_space=method_arguments.config_space,
                    search_options=search_options(method_arguments),
                    resource_attr=method_arguments.resource_attr,
                ),
            ),
            kwargs,
        )
    )


def MOBSTER(method_arguments: MethodArguments, **kwargs):
    return _MOBSTER(
        **recursive_merge(
            default_arguments(
                method_arguments,
                dict(
                    config_space=method_arguments.config_space,
                    search_options=search_options(method_arguments),
                    resource_attr=method_arguments.resource_attr,
                ),
            ),
            kwargs,
        )
    )


def BOHB(method_arguments: MethodArguments, **kwargs):
    return _BOHB(
        **recursive_merge(
            default_arguments(
                method_arguments,
                dict(
                    config_space=method_arguments.config_space,
                    search_options=search_options(method_arguments),
                    resource_attr=method_arguments.resource_attr,
                ),
            ),
            kwargs,
        )
    )


def SyncHyperband(method_arguments: MethodArguments, **kwargs):
    return _SyncHyperband(
        **recursive_merge(
            default_arguments(
                method_arguments,
                dict(
                    config_space=method_arguments.config_space,
                    search_options=search_options(method_arguments),
                    resource_attr=method_arguments.resource_attr,
                ),
            ),
            kwargs,
        )
    )


def SyncBOHB(method_arguments: MethodArguments, **kwargs):
    return _SyncBOHB(
        **recursive_merge(
            default_arguments(
                method_arguments,
                dict(
                    config_space=method_arguments.config_space,
                    search_options=search_options(method_arguments),
                    resource_attr=method_arguments.resource_attr,
                ),
            ),
            kwargs,
        )
    )


def DEHB(method_arguments: MethodArguments, **kwargs):
    return _DEHB(
        **recursive_merge(
            default_arguments(
                method_arguments,
                dict(
                    config_space=method_arguments.config_space,
                    search_options=search_options(method_arguments),
                    resource_attr=method_arguments.resource_attr,
                ),
            ),
            kwargs,
        )
    )


def SyncMOBSTER(method_arguments: MethodArguments, **kwargs):
    return _SyncMOBSTER(
        **recursive_merge(
            default_arguments(
                method_arguments,
                dict(
                    config_space=method_arguments.config_space,
                    search_options=search_options(method_arguments),
                    resource_attr=method_arguments.resource_attr,
                ),
            ),
            kwargs,
        )
    )


def KDE(method_arguments: MethodArguments, **kwargs):
    return _KDE(
        **recursive_merge(
            default_arguments(
                method_arguments,
                dict(
                    config_space=method_arguments.config_space,
                    search_options=search_options(method_arguments),
                ),
            ),
            kwargs,
        )
    )
