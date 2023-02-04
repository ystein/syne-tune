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
    convert_categorical_to_ordinal,
    convert_categorical_to_ordinal_numeric,
    search_options,
    default_arguments,
)
from syne_tune.optimizer.schedulers.hyperband import HyperbandScheduler
from syne_tune.optimizer.schedulers.synchronous import (
    SynchronousGeometricHyperbandScheduler,
    GeometricDifferentialEvolutionHyperbandScheduler,
)


class Methods:
    ASHA = "ASHA"
    SYNCHB = "SYNCHB"
    DEHB = "DEHB"
    BOHB = "BOHB"
    ASHA_ORD = "ASHA-ORD"
    SYNCHB_ORD = "SYNCHB-ORD"
    DEHB_ORD = "DEHB-ORD"
    BOHB_ORD = "BOHB-ORD"
    ASHA_STOP = "ASHA-STOP"
    SYNCMOBSTER = "SYNCMOBSTER"


def conv_numeric_only(margs) -> Dict[str, Any]:
    return convert_categorical_to_ordinal_numeric(
        margs.config_space, kind=margs.fcnet_ordinal
    )


def conv_numeric_then_rest(margs) -> Dict[str, Any]:
    return convert_categorical_to_ordinal(
        convert_categorical_to_ordinal_numeric(
            margs.config_space, kind=margs.fcnet_ordinal
        )
    )


methods = {
    Methods.ASHA: lambda method_arguments: HyperbandScheduler(
        **default_arguments(method_arguments),
        config_space=conv_numeric_only(method_arguments),
        searcher="random",
        type="promotion",
        search_options=search_options(method_arguments),
        resource_attr=method_arguments.resource_attr,
    ),
    Methods.SYNCHB: lambda method_arguments: SynchronousGeometricHyperbandScheduler(
        **default_arguments(method_arguments),
        config_space=conv_numeric_only(method_arguments),
        searcher="random",
        search_options=search_options(method_arguments),
        resource_attr=method_arguments.resource_attr,
    ),
    Methods.DEHB: lambda method_arguments: GeometricDifferentialEvolutionHyperbandScheduler(
        **default_arguments(method_arguments),
        config_space=conv_numeric_only(method_arguments),
        searcher="random_encoded",
        search_options=search_options(method_arguments),
        resource_attr=method_arguments.resource_attr,
    ),
    Methods.BOHB: lambda method_arguments: SynchronousGeometricHyperbandScheduler(
        **default_arguments(method_arguments),
        config_space=conv_numeric_only(method_arguments),
        searcher="kde",
        search_options=search_options(method_arguments),
        resource_attr=method_arguments.resource_attr,
    ),
    Methods.ASHA_ORD: lambda method_arguments: HyperbandScheduler(
        **default_arguments(method_arguments),
        config_space=conv_numeric_then_rest(method_arguments),
        searcher="random",
        type="promotion",
        search_options=search_options(method_arguments),
        resource_attr=method_arguments.resource_attr,
    ),
    Methods.SYNCHB_ORD: lambda method_arguments: SynchronousGeometricHyperbandScheduler(
        **default_arguments(method_arguments),
        config_space=conv_numeric_then_rest(method_arguments),
        searcher="random",
        search_options=search_options(method_arguments),
        resource_attr=method_arguments.resource_attr,
    ),
    Methods.DEHB_ORD: lambda method_arguments: GeometricDifferentialEvolutionHyperbandScheduler(
        **default_arguments(method_arguments),
        config_space=conv_numeric_then_rest(method_arguments),
        searcher="random_encoded",
        search_options=search_options(method_arguments),
        resource_attr=method_arguments.resource_attr,
    ),
    Methods.BOHB_ORD: lambda method_arguments: SynchronousGeometricHyperbandScheduler(
        **default_arguments(method_arguments),
        config_space=conv_numeric_then_rest(method_arguments),
        searcher="kde",
        search_options=search_options(method_arguments),
        resource_attr=method_arguments.resource_attr,
    ),
    Methods.ASHA_STOP: lambda method_arguments: HyperbandScheduler(
        **default_arguments(method_arguments),
        config_space=conv_numeric_only(method_arguments),
        searcher="random",
        type="stopping",
        search_options=search_options(method_arguments),
        resource_attr=method_arguments.resource_attr,
    ),
    Methods.SYNCMOBSTER: lambda method_arguments: SynchronousGeometricHyperbandScheduler(
        **default_arguments(method_arguments),
        config_space=conv_numeric_only(method_arguments),
        searcher="bayesopt",
        search_options=search_options(method_arguments),
        resource_attr=method_arguments.resource_attr,
    ),
}
