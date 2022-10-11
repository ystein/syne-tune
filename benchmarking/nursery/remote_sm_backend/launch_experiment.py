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
import logging
from argparse import ArgumentParser

from syne_tune.try_import import try_import_aws_message

try:
    from sagemaker.pytorch import PyTorch
except ImportError:
    print(try_import_aws_message())

from syne_tune.backend import SageMakerBackend
from syne_tune.backend.sagemaker_backend.sagemaker_utils import (
    get_execution_role,
    default_sagemaker_session,
)
from syne_tune.optimizer.baselines import (
    ASHA,
    RandomSearch,
    BayesianOptimization,
    MOBSTER,
)
from syne_tune.stopping_criterion import StoppingCriterion
from syne_tune.tuner import Tuner
from benchmarking.commons.benchmark_real_definitions import (
    benchmark_definitions,
)
from syne_tune.util import repository_root_path


def parse_args(launch_remote: bool = False):
    parser = ArgumentParser()
    parser.add_argument(
        "--benchmark",
        type=str,
        default="resnet_cifar10",
        help="benchmark to run",
    )
    parser.add_argument(
        "--experiment_tag",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        required=False,
        help="number of parallel workers",
    )
    parser.add_argument(
        "--max_wallclock_time",
        type=int,
        required=False,
        help="maximum wallclock time of experiment",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["ASHA", "MOBSTER", "RS", "BO"],
        default="ASHA",
        help="method to run",
    )
    parser.add_argument(
        "--max_failures",
        type=int,
        default=3,
        help="number of trials which can fail without experiment being terminated",
    )
    if launch_remote:
        parser.add_argument(
            "--num_seeds",
            type=int,
            required=True,
            help="number of seeds to run",
        )
        parser.add_argument(
            "--start_seed",
            type=int,
            default=0,
            help="first seed to run",
        )
    else:
        parser.add_argument(
            "--seed",
            type=int,
            default=0,
            help="seed (for repetitions)",
        )
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    args = parse_args()
    experiment_tag = args.experiment_tag
    benchmark_name = args.benchmark

    benchmark_kwargs = (
        dict(n_workers=args.n_workers) if args.n_workers is not None else dict()
    )
    benchmark = benchmark_definitions(sagemaker_backend=True, **benchmark_kwargs)[
        benchmark_name
    ]
    assert (
        benchmark.framework == "PyTorch"
    ), "Only support PyTorch estimator at the moment"
    if args.max_wallclock_time is not None:
        max_wallclock_time = args.max_wallclock_time
    else:
        max_wallclock_time = benchmark.max_wallclock_time

    print(f"Starting experiment ({args.seed}) of {experiment_tag}")

    script_path = benchmark.script
    estimator_kwargs = (
        benchmark.estimator_kwargs if benchmark.estimator_kwargs is not None else dict()
    )
    trial_backend = SageMakerBackend(
        # we tune a PyTorch Framework from Sagemaker
        sm_estimator=PyTorch(
            entry_point=script_path.name,
            source_dir=str(script_path.parent),
            instance_type=benchmark.instance_type,
            instance_count=1,
            role=get_execution_role(),
            max_run=int(1.2 * max_wallclock_time),
            dependencies=[str(repository_root_path() / "benchmarking/")],
            disable_profiler=True,
            debugger_hook_config=False,
            sagemaker_session=default_sagemaker_session(),
            **estimator_kwargs,
        ),
        # names of metrics to track. Each metric will be detected by Sagemaker if it is written in the
        # following form: "[RMSE]: 1.2", see in train_main_example how metrics are logged for an example
        metrics_names=[benchmark.metric],
    )

    common_kwargs = dict(
        config_space=benchmark.config_space,
        search_options={"debug_log": True},
        metric=benchmark.metric,
        mode=benchmark.mode,
        max_resource_attr=benchmark.max_resource_attr,
        random_seed=args.seed,
    )
    if args.method == "asha":
        scheduler = ASHA(
            **common_kwargs,
            resource_attr=benchmark.resource_attr,
        )
    elif args.method == "mobster":
        scheduler = MOBSTER(
            **common_kwargs,
            resource_attr=benchmark.resource_attr,
        )
    elif args.method == "rs":
        scheduler = RandomSearch(**common_kwargs)
    else:
        assert args.method == "bo"
        scheduler = BayesianOptimization(**common_kwargs)

    stop_criterion = StoppingCriterion(
        max_wallclock_time=max_wallclock_time,
        max_num_evaluations=benchmark.max_num_evaluations,
    )
    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=benchmark.n_workers,
        sleep_time=5.0,
        max_failures=args.max_failures,
        tuner_name=experiment_tag,
        metadata={
            "seed": args.seed,
            "algorithm": args.method,
            "type": "stopping",
            "tag": experiment_tag,
            "benchmark": benchmark_name,
            "n_workers": benchmark.n_workers,
            "max_wallclock_time": max_wallclock_time,
        },
    )

    tuner.run()
