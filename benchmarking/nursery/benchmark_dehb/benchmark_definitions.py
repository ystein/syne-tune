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
from typing import Optional

from benchmarking.nursery.benchmark_automl.benchmark_definitions import (
    BenchmarkDefinition,
    fcnet_benchmark,
    lcbench_benchmark,
)
from syne_tune.config_space import choice, Categorical


NAS201_MAX_WALLCLOCK_TIME = {
    "cifar10": 5 * 3600,
    "cifar100": 6 * 3600,
    "ImageNet16-120": 8 * 3600,
}


NAS201_N_WORKERS = {
    "cifar10": 4,
    "cifar100": 4,
    "ImageNet16-120": 8,
}


def _modify_ordering_categories(new_categories: Optional[list]):
    def modify_config_space(config_space: dict) -> dict:
        new_config_space = dict()
        new_set = set(new_categories)
        for name, domain in config_space.items():
            if isinstance(domain, Categorical):
                assert (
                    set(domain.categories) == new_set
                ), f"{name}: categories = {set(domain.categories)} != {new_set}"
                domain = choice(new_categories)
            new_config_space[name] = domain
        return new_config_space

    if new_categories is not None:
        return modify_config_space
    else:
        return None


def nas201_benchmark(dataset_name, new_categories: Optional[list] = None):
    """
    :param dataset_name: Name of dataset for NASBench-201
    :param new_categories: If given, the original config space is modified
        by changing each categorical domain to have this list of values. Must
        be a permutation of the original one:
        ["avg_pool_3x3", "nor_conv_3x3", "skip_connect", "nor_conv_1x1", "none"]
    """
    return BenchmarkDefinition(
        max_wallclock_time=NAS201_MAX_WALLCLOCK_TIME[dataset_name],
        n_workers=NAS201_N_WORKERS[dataset_name],
        elapsed_time_attr="metric_elapsed_time",
        metric="metric_valid_error",
        mode="min",
        blackbox_name="nasbench201",
        dataset_name=dataset_name,
        max_resource_attr="epochs",
        modify_config_space=_modify_ordering_categories(new_categories),
    )


# HIER! Use new_categories if ordering to be changed!

benchmark_definitions = {
    "fcnet-protein": fcnet_benchmark("protein_structure"),
    "fcnet-naval": fcnet_benchmark("naval_propulsion"),
    "fcnet-parkinsons": fcnet_benchmark("parkinsons_telemonitoring"),
    "fcnet-slice": fcnet_benchmark("slice_localization"),
    "nas201-cifar10": nas201_benchmark("cifar10"),
    "nas201-cifar100": nas201_benchmark("cifar100"),
    "nas201-ImageNet16-120": nas201_benchmark("ImageNet16-120"),
}

# 5 most expensive lcbench datasets
lc_bench_datasets = [
    "Fashion-MNIST",
    "airlines",
    "albert",
    "covertype",
    "christine",
]
for task in lc_bench_datasets:
    benchmark_definitions[
        "lcbench-" + task.replace("_", "-").replace(".", "")
    ] = lcbench_benchmark(task, datasets=lc_bench_datasets)
