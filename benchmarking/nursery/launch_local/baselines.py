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
)
from syne_tune.optimizer.baselines import (
    RandomSearch,
    BayesianOptimization,
    ASHA,
    MOBSTER,
)


class Methods:
    RS = "RS"
    BO = "BO"
    ASHA = "ASHA"
    MOBSTER = "MOBSTER"


methods = {
    Methods.RS: lambda method_arguments: RandomSearch(
        **default_arguments(method_arguments),
        config_space=method_arguments.config_space,
        search_options=search_options(method_arguments),
    ),
    Methods.BO: lambda method_arguments: BayesianOptimization(
        **default_arguments(method_arguments),
        config_space=method_arguments.config_space,
        search_options=search_options(method_arguments),
    ),
    Methods.ASHA: lambda method_arguments: ASHA(
        **default_arguments(method_arguments),
        config_space=method_arguments.config_space,
        search_options=search_options(method_arguments),
        type="promotion",
        resource_attr=method_arguments.resource_attr,
    ),
    Methods.MOBSTER: lambda method_arguments: MOBSTER(
        **default_arguments(method_arguments),
        config_space=method_arguments.config_space,
        search_options=search_options(method_arguments),
        type="promotion",
        resource_attr=method_arguments.resource_attr,
    ),
}
