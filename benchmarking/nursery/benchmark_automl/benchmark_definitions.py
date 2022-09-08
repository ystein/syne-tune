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
from typing import Optional, List, Callable

from syne_tune.config_space import ordinal, Ordinal


@dataclass
class BenchmarkDefinition:
    """
    If `modify_config_space` is given, the original configuration
    space from the blackbox is modified by this mapping before
    passing it to the creation of the scheduler.
    """

    max_wallclock_time: float
    n_workers: int
    elapsed_time_attr: str
    metric: str
    mode: str
    blackbox_name: str
    dataset_name: str
    max_resource_attr: str
    max_num_evaluations: Optional[int] = None
    surrogate: Optional[str] = None
    surrogate_kwargs: Optional[dict] = None
    datasets: Optional[List[str]] = None
    modify_config_space: Optional[Callable[[dict], dict]] = None


def fcnet_benchmark(dataset_name):
    return BenchmarkDefinition(
        max_wallclock_time=1200,
        n_workers=4,
        elapsed_time_attr="metric_elapsed_time",
        metric="metric_valid_loss",
        mode="min",
        blackbox_name="fcnet",
        dataset_name=dataset_name,
        max_resource_attr="epochs",
    )


def _modify_ordering_categories(new_categories: Optional[list]):
    def modify_config_space(config_space: dict) -> dict:
        new_config_space = dict()
        new_set = set(new_categories)
        for name, domain in config_space.items():
            if isinstance(domain, Ordinal):
                assert (
                    set(domain.categories) == new_set
                ), f"{name}: categories = {set(domain.categories)} != {new_set}"
                domain = ordinal(new_categories)
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
        by changing each `ordinal` domain to have this list of values. Must
        be a permutation of the original one:
        ["avg_pool_3x3", "nor_conv_3x3", "skip_connect", "nor_conv_1x1", "none"]
    """
    return BenchmarkDefinition(
        max_wallclock_time=6 * 3600,
        n_workers=4,
        elapsed_time_attr="metric_elapsed_time",
        metric="metric_valid_error",
        mode="min",
        blackbox_name="nasbench201",
        dataset_name=dataset_name,
        max_resource_attr="epochs",
        modify_config_space=_modify_ordering_categories(new_categories),
    )


def lcbench_benchmark(dataset_name, datasets):
    return BenchmarkDefinition(
        max_wallclock_time=7200,
        n_workers=4,
        elapsed_time_attr="time",
        metric="val_accuracy",
        mode="max",
        blackbox_name="lcbench",
        dataset_name=dataset_name,
        surrogate="KNeighborsRegressor",
        surrogate_kwargs={"n_neighbors": 1},
        max_num_evaluations=4000,
        datasets=datasets,
        max_resource_attr="epochs",
    )


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
