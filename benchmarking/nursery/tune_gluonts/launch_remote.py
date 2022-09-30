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
import itertools
from tqdm import tqdm

from sagemaker.mxnet import MXNet

import syne_tune
from syne_tune.backend.sagemaker_backend.sagemaker_utils import (
    get_execution_role,
)
from syne_tune.util import s3_experiment_path, random_string
from benchmarking.nursery.tune_gluonts.hpo_main import parse_args


if __name__ == "__main__":
    random_seed_offset = 31415627

    args = parse_args(remote=True)
    # Useful if not all experiments could be started:
    skip_initial_experiments = 0

    # Compare different HPO algorithms
    if args.optimizer is not None:
        optimizers = [args.optimizer]
    else:
        optimizers = ["rs", "bo", "asha", "mobster"]
    run_ids = list(range(args.num_seeds))
    num_experiments = len(optimizers) * len(run_ids)

    # Loop over all combinations and repetitions
    suffix = random_string(4)
    combinations = list(itertools.product(optimizers, run_ids))
    for exp_id, (optimizer, run_id) in tqdm(enumerate(combinations)):
        if exp_id < skip_initial_experiments:
            continue
        print(f"Experiment {exp_id} (of {num_experiments})")
        # Make sure that results (on S3) are written to different subdirectories.
        # Otherwise, the SM training job will download many previous results at
        # start
        tuner_name = f"{optimizer}-{run_id}"
        # Results written to S3 under this path
        checkpoint_s3_uri = s3_experiment_path(
            experiment_name=args.experiment_name, tuner_name=tuner_name
        )
        print(f"Results stored to {checkpoint_s3_uri}")
        # We use a different seed for each `run_id`
        seed = (random_seed_offset + run_id) % (2**32)

        # Each experiment run is executed as SageMaker training job
        hyperparameters = {
            "dataset": args.dataset,
            "local_root": "/opt/ml/input/data",
            "max_runtime": args.max_runtime,
            "train_epochs": args.train_epochs,
            "n_workers": args.n_workers,
            "experiment_name": args.experiment_name,
            "store_logs_checkpoints_to_s3": int(args.store_logs_checkpoints_to_s3),
            "optimizer": optimizer,
            "seed": seed,
        }

        # Pass Syne Tune sources as dependencies
        source_dir = str(Path(__file__).parent)
        entry_point = "hpo_main.py"
        dependencies = syne_tune.__path__ + [source_dir]
        # GluonTS needs MXNet estimator
        est = MXNet(
            entry_point=entry_point,
            source_dir=source_dir,
            checkpoint_s3_uri=checkpoint_s3_uri,
            instance_type="ml.c5.2xlarge",
            instance_count=1,
            framework_version="1.7",
            py_version="py3",
            max_run=int(1.25 * args.max_runtime),
            role=get_execution_role(),
            dependencies=dependencies,
            disable_profiler=True,
            hyperparameters=hyperparameters,
        )

        job_name = f"{args.experiment_name}-{tuner_name}-{suffix}"
        print(f"Launching {job_name}")
        est.fit(wait=False, job_name=job_name)

    print(
        "\nLaunched all requested experiments. Once everything is done, use this "
        "command to sync result files from S3:\n"
        f"$ aws s3 sync {s3_experiment_path(experiment_name=args.experiment_name)}/ "
        f'~/syne-tune/{args.experiment_name}/ --exclude "*" '
        '--include "*metadata.json" --include "*results.csv.zip"'
    )
