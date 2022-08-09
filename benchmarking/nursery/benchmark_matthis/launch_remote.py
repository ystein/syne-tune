from pathlib import Path
from tqdm import tqdm

from sagemaker.pytorch import PyTorch

from benchmarking.nursery.benchmark_matthis.benchmark_main import parse_args
from syne_tune.backend.sagemaker_backend.sagemaker_utils import (
    get_execution_role,
)
import syne_tune
import benchmarking
from syne_tune.util import s3_experiment_path, random_string


def _is_expensive_method(method: str) -> bool:
    return method.startswith("MOBSTER") or method.startswith("HYPERTUNE")


if __name__ == "__main__":
    args, method_names, benchmark_names, seeds = parse_args()
    if len(benchmark_names) == 1:
        benchmark_name = benchmark_names[0]
    else:
        benchmark_name = None
    experiment_tag = args.experiment_tag
    suffix = random_string(4)
    combinations = []
    for method in method_names:
        seed_range = seeds if _is_expensive_method(method) else [None]
        combinations.extend([(method, seed) for seed in seed_range])
    for method, seed in tqdm(combinations):
        if seed is not None:
            tuner_name = f"{method}-{seed}"
        else:
            tuner_name = method
        checkpoint_s3_uri = s3_experiment_path(
            tuner_name=tuner_name, experiment_name=experiment_tag
        )
        sm_args = dict(
            entry_point="benchmark_main.py",
            source_dir=str(Path(__file__).parent),
            checkpoint_s3_uri=checkpoint_s3_uri,
            instance_type="ml.c5.4xlarge",
            instance_count=1,
            py_version="py38",
            framework_version="1.10.0",
            max_run=3600 * 72,
            role=get_execution_role(),
            dependencies=syne_tune.__path__ + benchmarking.__path__,
            disable_profiler=True,
        )
        hyperparameters = {
            "experiment_tag": experiment_tag,
            "method": method,
            "num_brackets": args.num_brackets,
            "num_samples": args.num_samples,
        }
        if seed is not None:
            hyperparameters["num_seeds"] = seed
            hyperparameters["run_all_seed"] = 0
        else:
            hyperparameters["num_seeds"] = args.num_seeds
            hyperparameters["start_seed"] = args.start_seed
        if benchmark_name is not None:
            hyperparameters["benchmark"] = benchmark_name
        max_t = getattr(args, "max_t")
        if max_t is not None:
            hyperparameters["max_t"] = max_t
        sm_args["hyperparameters"] = hyperparameters
        print(
            f"{experiment_tag}-{method}\n"
            f"hyperparameters = {hyperparameters}\n"
            f"Results written to {checkpoint_s3_uri}"
        )
        est = PyTorch(**sm_args)
        est.fit(job_name=f"{experiment_tag}-{tuner_name}-{suffix}", wait=False)

    print(
        "\nLaunched all requested experiments. Once everything is done, use this "
        "command to sync result files from S3:\n"
        f"$ aws s3 sync {s3_experiment_path(experiment_name=experiment_tag)}/ "
        f'~/syne-tune/{experiment_tag}/ --exclude "*" '
        '--include "*metadata.json" --include "*results.csv.zip"'
    )
