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
from dataclasses import dataclass
from typing import Optional, Dict
from pathlib import Path

from benchmarking.definitions.definition_resnet_cifar10 import (
    resnet_cifar10_benchmark as _resnet_cifar10_benchmark,
    resnet_cifar10_default_params,
)
from benchmarking.nursery.lstm_wikitext2.definition_lstm_wikitext2 import (
    lstm_wikitext2_benchmark as _lstm_wikitext2_benchmark,
    lstm_wikitext2_default_params,
)


@dataclass
class RealBenchmarkDefinition:
    script: Path
    config_space: dict
    max_wallclock_time: float
    n_workers: int
    instance_type: str
    metric: str
    mode: str
    max_resource_attr: str
    resource_attr: Optional[str] = None
    framework: Optional[str] = None
    estimator_kwargs: Optional[dict] = None
    max_num_evaluations: Optional[int] = None


def resnet_cifar10_benchmark(sagemaker_backend: bool = False, **kwargs):
    params = {"backend": "sagemaker"} if sagemaker_backend else None
    params = resnet_cifar10_default_params(params)
    benchmark = _resnet_cifar10_benchmark(params)
    _kwargs = dict(
        script=Path(benchmark["script"]),
        config_space=benchmark["config_space"],
        max_wallclock_time=3 * 3600,
        n_workers=params["num_workers"],
        instance_type=params["instance_type"],
        metric=benchmark["metric"],
        mode=benchmark["mode"],
        max_resource_attr=benchmark["max_resource_attr"],
        resource_attr=benchmark["resource_attr"],
        framework="PyTorch",
        estimator_kwargs=dict(
            framework_version="1.7.1",
            py_version="py3",
        ),
    )
    _kwargs.update(kwargs)
    return RealBenchmarkDefinition(**_kwargs)


def lstm_wikitext2_benchmark(sagemaker_backend: bool = False, **kwargs):
    params = {"backend": "sagemaker"} if sagemaker_backend else None
    params = lstm_wikitext2_default_params(params)
    benchmark = _lstm_wikitext2_benchmark(params)
    _kwargs = dict(
        script=Path(benchmark["script"]),
        config_space=benchmark["config_space"],
        max_wallclock_time=7 * 3600,
        n_workers=params["num_workers"],
        instance_type=params["instance_type"],
        metric=benchmark["metric"],
        mode=benchmark["mode"],
        max_resource_attr=benchmark["max_resource_attr"],
        resource_attr=benchmark["resource_attr"],
        framework="PyTorch",
        estimator_kwargs=dict(
            framework_version="1.7.1",
            py_version="py3",
        ),
    )
    _kwargs.update(kwargs)
    return RealBenchmarkDefinition(**_kwargs)


def benchmark_definitions(
    sagemaker_backend: bool = False, **kwargs
) -> Dict[str, RealBenchmarkDefinition]:
    return {
        "resnet_cifar10": resnet_cifar10_benchmark(sagemaker_backend, **kwargs),
        "lstm_wikitext2": lstm_wikitext2_benchmark(sagemaker_backend, **kwargs),
    }
