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
from typing import Optional, List
from pathlib import Path
from tqdm import tqdm
import itertools

from syne_tune.try_import import try_import_aws_message

try:
    from sagemaker.pytorch import PyTorch
except ImportError:
    print(try_import_aws_message())

from benchmarking.commons.hpo_local_main import parse_args
from benchmarking.commons.launch_remote import filter_none, message_sync_from_s3
from syne_tune.backend.sagemaker_backend.sagemaker_utils import (
    get_execution_role,
)
import syne_tune
import benchmarking
from syne_tune.util import s3_experiment_path, random_string


def launch_remote(
    entry_point: Path,
    methods: dict,
    benchmark_definitions: callable,
    extra_args: Optional[List[dict]] = None,
    map_extra_args: Optional[callable] = None,
):
    args, method_names, seeds = parse_args(methods, extra_args)
    benchmark_name = args.benchmark
    experiment_tag = args.experiment_tag
    suffix = random_string(4)
    benchmark_kwargs = (
        dict(n_workers=args.n_workers) if args.n_workers is not None else dict()
    )
    benchmark = benchmark_definitions(**benchmark_kwargs)[benchmark_name]
    assert (
        benchmark.framework == "PyTorch"
    ), "Only support PyTorch estimator at the moment"

    combinations = list(itertools.product(method_names, seeds))
    for method, seed in tqdm(combinations):
        tuner_name = f"{method}-{seed}"
        checkpoint_s3_uri = s3_experiment_path(
            tuner_name=tuner_name, experiment_name=experiment_tag
        )
        # TODO: What about dependencies of training script? Not sorted out
        # right now!
        sm_args = dict(
            entry_point=entry_point.name,
            source_dir=str(entry_point.parent),
            checkpoint_s3_uri=checkpoint_s3_uri,
            instance_type=benchmark.instance_type,
            instance_count=1,
            max_run=int(1.2 * benchmark.max_wallclock_time),
            role=get_execution_role(),
            dependencies=syne_tune.__path__ + benchmarking.__path__,
            disable_profiler=True,
            debugger_hook_config=False,
        )
        if benchmark.estimator_kwargs is not None:
            sm_args.update(benchmark.estimator_kwargs)

        hyperparameters = {
            "experiment_tag": experiment_tag,
            "benchmark": benchmark_name,
            "method": method,
            "save_tuner": int(args.save_tuner),
            "num_seeds": seed,
            "run_all_seed": 0,
            "verbose": int(args.verbose),
        }
        if args.n_workers is not None:
            hyperparameters["n_workers"] = args.n_workers
        if extra_args is not None:
            assert map_extra_args is not None
            hyperparameters.update(filter_none(map_extra_args(args)))
        sm_args["hyperparameters"] = hyperparameters
        print(
            f"{experiment_tag}-{tuner_name}\n"
            f"hyperparameters = {hyperparameters}\n"
            f"Results written to {checkpoint_s3_uri}"
        )
        est = PyTorch(**sm_args)
        est.fit(job_name=f"{experiment_tag}-{tuner_name}-{suffix}", wait=False)

    print("\n" + message_sync_from_s3(experiment_tag))
