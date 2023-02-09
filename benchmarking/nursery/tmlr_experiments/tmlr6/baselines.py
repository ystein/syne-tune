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
    MOBSTER,
    HyperTune,
    BOHB,
    ASHABORE,
    SyncBOHB,
    DEHB,
)


class Methods:
    MOBSTER_INDEP = "MOBSTER-INDEP"
    MOBSTER_M52_RW = "MOBSTER-M52-RW"
    HYPERTUNE_JOINT = "HyperTune-JOINT"
    BOHB = "BOHB"
    ASHABORE = "ASHABORE"
    SYNCBOHB = "SyncBOHB"
    DEHB = "DEHB"


methods = {
    Methods.MOBSTER_INDEP: lambda method_arguments: MOBSTER(
        method_arguments,
        type="promotion",
        search_options=dict(model="gp_independent"),
    ),
    Methods.MOBSTER_M52_RW: lambda method_arguments: MOBSTER(
        method_arguments,
        type="promotion",
        search_options=dict(gp_resource_kernel="matern52-res-warp"),
    ),
    Methods.HYPERTUNE_JOINT: lambda method_arguments: HyperTune(
        method_arguments,
        type="promotion",
        search_options=dict(model="gp_multitask"),
    ),
    Methods.BOHB: lambda method_arguments: BOHB(method_arguments, type="promotion"),
    Methods.ASHABORE: lambda method_arguments: ASHABORE(
        method_arguments, type="promotion"
    ),
    Methods.SYNCBOHB: lambda method_arguments: SyncBOHB(method_arguments),
    Methods.DEHB: lambda method_arguments: DEHB(method_arguments),
}
