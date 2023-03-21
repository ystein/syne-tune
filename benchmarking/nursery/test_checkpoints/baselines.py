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
)


class Methods:
    ASHA = "ASHA"
    ASHA_RCP_DEF = "ASHA-RCP-DEF"
    ASHA_RCP_VAR1 = "ASHA-RCP-VAR1"
    ASHA_RCP_VAR2 = "ASHA-RCP-VAR2"
    ASHA_RCP_VAR3 = "ASHA-RCP-VAR3"


methods = {
    # Methods.ASHA: lambda method_arguments: ASHA(method_arguments, type="promotion"),
    Methods.ASHA_RCP_DEF: lambda method_arguments: ASHA(
        method_arguments,
        type="promotion",
        early_checkpoint_removal_kwargs=dict(
            max_num_checkpoints=10,
        ),
    ),
    Methods.ASHA_RCP_VAR1: lambda method_arguments: ASHA(
        method_arguments,
        type="promotion",
        early_checkpoint_removal_kwargs=dict(
            max_num_checkpoints=10,
            prior_beta_mean=0.15,
        ),
    ),
    Methods.ASHA_RCP_VAR2: lambda method_arguments: ASHA(
        method_arguments,
        type="promotion",
        early_checkpoint_removal_kwargs=dict(
            max_num_checkpoints=10,
            prior_beta_size=0.5,
        ),
    ),
    Methods.ASHA_RCP_VAR3: lambda method_arguments: ASHA(
        method_arguments,
        type="promotion",
        early_checkpoint_removal_kwargs=dict(
            max_num_checkpoints=10,
            min_data_at_rung=10,
        ),
    ),
}
