from argparse import ArgumentParser
from pathlib import Path

from coolname import generate_slug
from sagemaker.pytorch import PyTorch

from benchmarking.nursery.benchmark_kdd.baselines import methods, Methods
from syne_tune.backend.sagemaker_backend.sagemaker_utils import get_execution_role
import syne_tune
import benchmarking
from syne_tune.util import s3_experiment_path, random_string


if __name__ == '__main__':
    num_seeds = 30
    parser = ArgumentParser()
    parser.add_argument("--experiment_tag", type=str, required=False, default=generate_slug(2))
    args, _ = parser.parse_known_args()
    experiment_tag = args.experiment_tag
    hash = random_string(4)
    for method in ['ZS', 'RUSH', 'RUSH5', 'ASHA-BB', 'ASHA-CTS', 'RS', 'ASHA', 'BOHB', 'REA', 'TPE']:  # methods.keys():
        sm_args = dict(
            entry_point="benchmark_main.py",
            source_dir=str(Path(__file__).parent),
            # instance_type="local",
            checkpoint_s3_uri=s3_experiment_path(tuner_name=experiment_tag),
            instance_type="ml.c5.4xlarge",
            instance_count=1,
            py_version="py3",
            framework_version='1.6',
            max_run=3600 * 72,
            role='arn:aws:iam::536276317016:role/AmazonSageMakerServiceCatalogProductsUseRole',
            dependencies=syne_tune.__path__ + benchmarking.__path__,
            disable_profiler=True,
        )
        if method not in [Methods.MOBSTER, Methods.SGPT]:
            print(f"{experiment_tag}-{method}")
            sm_args["hyperparameters"] = {"experiment_tag": experiment_tag, 'num_seeds': num_seeds, 'method': method}
            est = PyTorch(**sm_args)
            est.fit(job_name=f"{experiment_tag}-{method}-{hash}", wait=False)
        else:
            # For mobster and sgpt, we schedule one job per seed as the method takes much longer
            for seed in range(num_seeds):
                print(f"{experiment_tag}-{method}-{seed}")
                sm_args["hyperparameters"] = {
                    "experiment_tag": experiment_tag, 'num_seeds': seed, 'run_all_seed': 0,
                    'method': method
                }
                est = PyTorch(**sm_args)
                est.fit(job_name=f"{experiment_tag}-{method}-{seed}-{hash}", wait=False)
