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
from tqdm import tqdm
import os

from syne_tune.try_import import try_import_aws_message

try:
    import boto3
    from sagemaker.pytorch import PyTorch
except ImportError:
    print(try_import_aws_message())

from syne_tune.backend.sagemaker_backend.sagemaker_utils import (
    get_execution_role,
)
import syne_tune
import benchmarking
from syne_tune.util import s3_experiment_path, random_string
from benchmarking.commons.launch_remote import message_sync_from_s3
from benchmarking.commons.benchmark_real_definitions import (
    benchmark_definitions,
)
from benchmarking.nursery.remote_sm_backend.launch_experiment import parse_args


if __name__ == "__main__":
    args = parse_args(launch_remote=True)
    experiment_tag = args.experiment_tag
    suffix = random_string(4)

    if boto3.Session().region_name is None:
        os.environ["AWS_DEFAULT_REGION"] = "us-west-2"
    environment = {"AWS_DEFAULT_REGION": boto3.Session().region_name}
    benchmark = benchmark_definitions(sagemaker_backend=True)[args.benchmark]
    assert (
        benchmark.framework == "PyTorch"
    ), "Only support PyTorch estimator at the moment"
    if args.max_wallclock_time is not None:
        max_wallclock_time = args.max_wallclock_time
    else:
        max_wallclock_time = benchmark.max_wallclock_time

    for seed in tqdm(range(args.start_seed, args.num_seeds)):
        tuner_name = f"{args.method}-{seed}"
        checkpoint_s3_uri = s3_experiment_path(
            tuner_name=tuner_name, experiment_name=experiment_tag
        )
        sm_args = dict(
            entry_point="launch_experiment.py",
            source_dir=str(Path(__file__).parent),
            checkpoint_s3_uri=checkpoint_s3_uri,
            instance_type="ml.c5.4xlarge",
            instance_count=1,
            py_version="py38",
            framework_version="1.10.0",
            max_run=int(1.2 * max_wallclock_time),
            role=get_execution_role(),
            dependencies=syne_tune.__path__ + benchmarking.__path__,
            disable_profiler=True,
            debugger_hook_config=False,
            environment=environment,
        )
        hyperparameters = {
            "benchmark": args.benchmark,
            "experiment_tag": experiment_tag,
            "seed": seed,
            "method": args.method,
            "max_failures": args.max_failures,
        }
        if args.n_workers is not None:
            hyperparameters["n_workers"] = args.n_workers
        if args.max_wallclock_time is not None:
            hyperparameters["max_wallclock_time"] = args.max_wallclock_time
        sm_args["hyperparameters"] = hyperparameters
        print(
            f"{experiment_tag}-{tuner_name}\n"
            f"hyperparameters = {hyperparameters}\n"
            f"Results written to {checkpoint_s3_uri}"
        )
        est = PyTorch(**sm_args)
        est.fit(job_name=f"{experiment_tag}-{tuner_name}-{suffix}", wait=False)

    print("\n" + message_sync_from_s3(experiment_tag))
