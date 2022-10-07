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
from typing import Optional, List, Callable

import numpy as np
import itertools
import logging
from argparse import ArgumentParser
from tqdm import tqdm
import copy

try:
    from coolname import generate_slug
except ImportError:
    print("coolname is not installed, will not be used")

from syne_tune.backend import LocalBackend
from syne_tune.stopping_criterion import StoppingCriterion
from syne_tune.tuner import Tuner
from benchmarking.commons.baselines import MethodArguments


def parse_args(methods: dict, extra_args: Optional[List[dict]] = None):
    try:
        default_experiment_tag = generate_slug(2)
    except Exception:
        default_experiment_tag = "syne_tune_experiment"
    parser = ArgumentParser()
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        help="Benchmark to run",
    )
    parser.add_argument(
        "--experiment_tag",
        type=str,
        default=default_experiment_tag,
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=30,
        help="number of seeds to run",
    )
    parser.add_argument(
        "--run_all_seeds",
        type=int,
        default=1,
        help="if 1 run all the seeds [0, `num_seeds`-1], otherwise run seed `num_seeds` only",
    )
    parser.add_argument(
        "--start_seed",
        type=int,
        default=0,
        help="first seed to run (if `run_all_seed` == 1)",
    )
    parser.add_argument(
        "--method", type=str, required=False, help="a method to run from baselines.py"
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        help="verbose log output?",
    )
    parser.add_argument(
        "--save_tuner",
        type=int,
        default=0,
        help="Serialize Tuner object at the end of tuning?",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        required=False,
        help="Number of workers",
    )
    if extra_args is not None:
        extra_args = copy.deepcopy(extra_args)
        for kwargs in extra_args:
            name = kwargs.pop("name")
            parser.add_argument(name, **kwargs)
    args, _ = parser.parse_known_args()
    args.verbose = bool(args.verbose)
    args.save_tuner = bool(args.save_tuner)
    args.run_all_seeds = bool(args.run_all_seeds)
    if args.run_all_seeds:
        seeds = list(range(args.start_seed, args.num_seeds))
    else:
        seeds = [args.num_seeds]
    method_names = [args.method] if args.method is not None else list(methods.keys())
    return args, method_names, seeds


def main(
    methods: dict,
    benchmark_definitions: callable,
    extra_args: Optional[List[dict]] = None,
    map_extra_args: Optional[Callable] = None,
):
    args, method_names, seeds = parse_args(methods, extra_args)
    experiment_tag = args.experiment_tag
    benchmark_name = args.benchmark

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger("syne_tune.optimizer.schedulers").setLevel(logging.WARNING)
        logging.getLogger("syne_tune.backend").setLevel(logging.WARNING)
        logging.getLogger(
            "syne_tune.backend.simulator_backend.simulator_backend"
        ).setLevel(logging.WARNING)
    benchmark_kwargs = (
        dict(n_workers=args.n_workers) if args.n_workers is not None else dict()
    )
    benchmark = benchmark_definitions(**benchmark_kwargs)[benchmark_name]

    combinations = list(itertools.product(method_names, seeds))
    print(combinations)
    for method, seed in tqdm(combinations):
        np.random.seed(seed)
        print(
            f"Starting experiment ({method}/{benchmark_name}/{seed}) of {experiment_tag}"
        )
        trial_backend = LocalBackend(entry_point=str(benchmark.script))

        method_kwargs = {"max_resource_attr": benchmark.max_resource_attr}
        if extra_args is not None:
            assert map_extra_args is not None
            extra_args = map_extra_args(args)
            method_kwargs.update(extra_args)
        scheduler = methods[method](
            MethodArguments(
                config_space=benchmark.config_space,
                metric=benchmark.metric,
                mode=benchmark.mode,
                random_seed=seed,
                resource_attr=benchmark.resource_attr,
                verbose=args.verbose,
                **method_kwargs,
            )
        )

        stop_criterion = StoppingCriterion(
            max_wallclock_time=benchmark.max_wallclock_time,
            max_num_evaluations=benchmark.max_num_evaluations,
        )
        metadata = {
            "seed": seed,
            "algorithm": method,
            "tag": experiment_tag,
            "benchmark": benchmark_name,
        }
        if args.n_workers is not None:
            metadata["n_workers"] = args.n_workers
        if extra_args is not None:
            metadata.update(extra_args)
        tuner = Tuner(
            trial_backend=trial_backend,
            scheduler=scheduler,
            stop_criterion=stop_criterion,
            n_workers=benchmark.n_workers,
            tuner_name=experiment_tag,
            metadata=metadata,
            save_tuner=args.save_tuner,
        )
        tuner.run()
