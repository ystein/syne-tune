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
    ASHA,
    MOBSTER,
    SyncHyperband,
    SyncMOBSTER,
)


class Methods:
    BO = "BO"
    ASHA = "ASHA"
    MOBSTER = "MOBSTER"
    SYNCHB = "SyncHB"
    SYNCMOBSTER = "SyncMOBSTER"


methods = {
    Methods.BO: lambda method_arguments: BayesianOptimization(method_arguments),
    Methods.ASHA: lambda method_arguments: ASHA(method_arguments, type="promotion"),
    Methods.MOBSTER: lambda method_arguments: MOBSTER(
        method_arguments, type="promotion"
    ),
    Methods.SYNCHB: lambda method_arguments: SyncHyperband(method_arguments),
    Methods.SYNCMOBSTER: lambda method_arguments: SyncMOBSTER(method_arguments),
}
