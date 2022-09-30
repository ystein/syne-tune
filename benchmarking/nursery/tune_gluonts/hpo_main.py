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
import argparse
from pathlib import Path

from syne_tune.config_space import loguniform, lograndint
from syne_tune.backend import LocalBackend
from syne_tune.optimizer.baselines import (
    RandomSearch,
    BayesianOptimization,
    ASHA,
    MOBSTER,
)
from syne_tune import Tuner, StoppingCriterion


def parse_args(remote=False):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="electricity",
        choices=("electricity", "m4-hourly"),
    )
    parser.add_argument(
        "--local_root",
        type=str,
        default=str(Path("~").expanduser()),
    )
    parser.add_argument("--max_runtime", type=int, default=1800)
    parser.add_argument("--train_epochs", type=int, default=50)
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--experiment_name", type=str, default="gluonts-1")
    if not remote:
        parser.add_argument(
            "--optimizer",
            type=str,
            default="asha",
            choices=("rs", "bo", "asha", "mobster"),
        )
        parser.add_argument("--seed", type=int, default=0)
    else:
        parser.add_argument(
            "--optimizer",
            type=str,
            required=False,
            choices=("rs", "bo", "asha", "mobster"),
        )
        parser.add_argument("--num_seeds", type=int, default=1)

    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    args = parse_args()

    config_space = {
        "lr": loguniform(lower=1e-4, upper=1e-1),
        "num_cells": lograndint(lower=1, upper=80),
        "num_layers": lograndint(lower=1, upper=10),
        "epochs": args.train_epochs,
        "dataset": args.dataset,
        "dataset_path": str(Path(args.local_root) / ".mxnet" / "gluon-ts" / "datasets"),
    }
    default_config = {
        "lr": 0.001,
        "num_cells": 40,
        "num_layers": 2,
    }

    # The backend is responsible to start and stop training evaluations. Here, we
    # use the local backend, which runs on a single instance
    entry_point = str(Path(__file__).parent / "train_gluonts.py")
    trial_backend = LocalBackend(entry_point=entry_point)

    # HPO algorithm
    # We can choose from these optimizers:
    schedulers = {
        "rs": RandomSearch,
        "bo": BayesianOptimization,
        "asha": ASHA,
        "mobster": MOBSTER,
    }
    scheduler_kwargs = dict(
        metric="mean_wQuantileLoss",
        mode="min",
        random_seed=args.seed,
        points_to_evaluate=[default_config],  # evaluate this one first
    )
    if args.optimizer in {"asha", "mobster"}:
        # The multi-fidelity methods need extra information
        scheduler_kwargs["resource_attr"] = "epoch_no"
        # Maximum resource level information in `config_space`:
        scheduler_kwargs["max_resource_attr"] = "epochs"
    scheduler = schedulers[args.optimizer](config_space, **scheduler_kwargs)

    # All parts come together in the tuner, which runs the experiment
    stop_criterion = StoppingCriterion(max_wallclock_time=args.max_runtime)
    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=args.n_workers,
        metadata=vars(args),  # metadata is stored along with results
        tuner_name=args.experiment_name,
        # some failures may happen when SGD diverges with NaNs
        max_failures=10,
    )

    tuner.run()  # off we go!
