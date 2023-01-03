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
from benchmarking.commons.benchmark_definitions import (
    nas201_benchmark_definitions,
    yahpo_nb301_benchmark_definitions,
    fcnet_benchmark_definitions,
)


nb201_key = "nas201-cifar100"

# We use all four FCNet datasets here, to be safe, but only CIFAR100 for
# NB201
benchmark_definitions = {
    nb201_key: nas201_benchmark_definitions[nb201_key],
    **yahpo_nb301_benchmark_definitions,
    **fcnet_benchmark_definitions,
}
