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
from pathlib import Path

from benchmarking.commons.benchmark_definitions.common import RealBenchmarkDefinition
from benchmarking.nursery.odsc_tutorial.resnet_cifar10.code.resnet_cifar10 import (
    METRIC_NAME,
    RESOURCE_ATTR,
    MAX_RESOURCE_ATTR,
    _config_space,
)
from syne_tune.remote.estimators import (
    DEFAULT_GPU_INSTANCE_1GPU,
    DEFAULT_GPU_INSTANCE_4GPU,
)


def resnet_cifar10_benchmark(sagemaker_backend: bool = False, **kwargs):
    if sagemaker_backend:
        instance_type = DEFAULT_GPU_INSTANCE_1GPU
    else:
        # For local backend, GPU cores serve different workers
        instance_type = DEFAULT_GPU_INSTANCE_4GPU
    fixed_parameters = {
        MAX_RESOURCE_ATTR: 27,
        "dataset_path": "./",
        "num_gpus": 1,
    }
    config_space = {**_config_space, **fixed_parameters}
    _kwargs = dict(
        script=Path(__file__).parent / "resnet_cifar10.py",
        config_space=config_space,
        metric=METRIC_NAME,
        mode="max",
        max_resource_attr=MAX_RESOURCE_ATTR,
        resource_attr=RESOURCE_ATTR,
        max_wallclock_time=3 * 3600,
        n_workers=4,
        instance_type=instance_type,
        framework="PyTorch",
    )
    _kwargs.update(kwargs)
    return RealBenchmarkDefinition(**_kwargs)
