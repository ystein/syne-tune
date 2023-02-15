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
    ASHA,
    MOBSTER,
    HyperTune,
)


class Methods:
    ASHA_STOP = "ASHA-STOP"
    MOBSTER_STOP = "MOBSTER-STOP"
    HYPERTUNE_STOP = "HyperTune-STOP"


methods = {
    Methods.ASHA_STOP: lambda method_arguments: ASHA(method_arguments, type="stopping"),
    Methods.MOBSTER_STOP: lambda method_arguments: MOBSTER(
        method_arguments, type="stopping"
    ),
    Methods.HYPERTUNE_STOP: lambda method_arguments: HyperTune(
        method_arguments, type="stopping"
    ),
}
