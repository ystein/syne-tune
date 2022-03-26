from pathlib import Path
import itertools

from sagemaker.pytorch import PyTorch
# from sagemaker.huggingface import HuggingFace
import syne_tune

from syne_tune.backend.sagemaker_backend.sagemaker_utils import get_execution_role
from syne_tune.util import s3_experiment_path, name_from_base


if __name__ == '__main__':
    # Setup of experiments:
    # - Compare selecting the model to fixing it to a default choice
    # - Compare 4 different HPO algorithms
    # - Repeat each experiment 10 times
    experiment_name = 'glue-3'
    random_seed_offset = 31415627
    num_runs = 10
    optimizers = ['rs', 'bo', 'asha', 'mobster']
    model_selection = [False, True]
    run_ids = list(range(num_runs))
    num_experiments = len(model_selection) * len(optimizers) * len(run_ids)
    # Useful if not all experiments could be started:
    skip_initial_experiments = 0

    dataset_name = 'rte'  # GLUE task
    model_type = 'bert-base-cased'  # Default model used if not selected
    num_train_epochs = 3  # Maximum number of epochs
    max_runtime = 1800  # Each experiment runs for 30 mins
    instance_type = 'ml.g4dn.xlarge'
    # We need 1 GPU for each worker:
    if instance_type == 'ml.g4dn.12xlarge':
        n_workers = 4
    else:
        n_workers = 1

    first_tuner_name = None
    last_tuner_name = None
    for exp_id, (choose_model, optimizer, run_id) in enumerate(
            itertools.product(model_selection, optimizers, run_ids)):
        if exp_id < skip_initial_experiments:
            continue
        print(f"Experiment {exp_id} (of {num_experiments})")
        # Append date time-stamp postfix. This is done here (and not when `Tuner`
        # is created), so that the SM job name coincides with the path name where
        # results are stored
        tuner_name = name_from_base(experiment_name, default="st-tuner")
        if first_tuner_name is None:
            first_tuner_name = tuner_name
        last_tuner_name = tuner_name
        # Results written to S3 under this path
        # Note: The S3 path has `tuner_name` as postfix, so that only files
        # related to this particular experiment run are covered by the sync
        # mechanism. This is NOT the default, and we need further changes in
        # `hpo_main.py`.
        checkpoint_s3_uri = s3_experiment_path(
            experiment_name=experiment_name, tuner_name=tuner_name)
        print(f"Results stored to {checkpoint_s3_uri}")
        # We use a different seed for each `run_id`
        seed = (random_seed_offset + run_id) % (2 ** 32)

        # Each experiment is executed as SageMaker training job
        entry_point = 'hpo_main.py'
        hyperparameters = {
            'run_id': run_id,
            'dataset': dataset_name,
            'model_type': model_type,
            'max_runtime': max_runtime,
            "num_train_epochs": num_train_epochs,
            'n_workers': n_workers,
            'optimizer': optimizer,
            'tuner_name': tuner_name,
            'choose_model': choose_model,
            'seed': seed,
        }

        # Pass Syne Tune sources as dependencies
        source_dir = str(Path(__file__).parent)
        dependencies = syne_tune.__path__ + [source_dir]
        #est = HuggingFace(
        #    entry_point=entry_point,
        #    source_dir=source_dir,
        #    checkpoint_s3_uri=checkpoint_s3_uri,
        #    instance_type=instance_type,
        #    instance_count=1,
        #    py_version="py38",
        #    pytorch_version='1.9',
        #    transformers_version='4.12',
        #    volume_size=125,
        #    max_run=int(1.25 * max_runtime),
        #    role=get_execution_role(),
        #    dependencies=dependencies,
        #    disable_profiler=True,
        #    hyperparameters=hyperparameters,
        #)
        # Latest PyTorch version:
        est = PyTorch(
            entry_point=entry_point,
            source_dir=source_dir,
            checkpoint_s3_uri=checkpoint_s3_uri,
            instance_type=instance_type,
            instance_count=1,
            py_version="py38",
            framework_version='1.10',
            volume_size=125,
            max_run=int(1.25 * max_runtime),
            role=get_execution_role(),
            dependencies=dependencies,
            disable_profiler=True,
            hyperparameters=hyperparameters,
        )
        # Old PyTorch version (worked for Aaron,
        # but "torch=1.10.0" in requirements.txt)
        #est = PyTorch(
        #    entry_point=entry_point,
        #    source_dir=source_dir,
        #    checkpoint_s3_uri=checkpoint_s3_uri,
        #    instance_type=instance_type,
        #    instance_count=1,
        #    py_version="py3",
        #    framework_version='1.6',
        #    volume_size = 125,
        #    max_run=int(1.25 * max_runtime),
        #    role=get_execution_role(),
        #    dependencies=dependencies,
        #    disable_profiler=True,
        #    hyperparameters=hyperparameters,
        #)

        print(f"Launching choose_model = {choose_model}, optimizer = {optimizer}, "
              f"run_id = {run_id} as {tuner_name}")
        est.fit(wait=False, job_name=tuner_name)
        #if exp_id == 0:
        #    break

    print(f"\nFor the record:\n{first_tuner_name} .. {last_tuner_name}")
