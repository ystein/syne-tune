import logging
import argparse
from pathlib import Path

from syne_tune.backend.local_backend import LocalBackend
from syne_tune.tuner import Tuner
from syne_tune.config_space import uniform, loguniform, choice, randint
from syne_tune.stopping_criterion import StoppingCriterion
from syne_tune.optimizer.baselines import ASHA, MOBSTER, BayesianOptimization, \
    RandomSearch
# from syne_tune.optimizer.schedulers.searchers.searcher_callback import \
#     StoreResultsAndModelParamsCallback


# Different GLUE tasks and their metric names
TASK2METRICSMODE = {
    "cola": {'metric': 'matthews_correlation', 'mode': 'max'},
    "mnli": {'metric': 'accuracy', 'mode': 'max'},
    "mrpc": {'metric': 'f1', 'mode': 'max'},
    "qnli": {'metric': 'accuracy', 'mode': 'max'},
    "qqp": {'metric': 'f1', 'mode': 'max'},
    "rte": {'metric': 'accuracy', 'mode': 'max'},
    "sst2": {'metric': 'accuracy', 'mode': 'max'},
    "stsb": {'metric': 'spearmanr', 'mode': 'max'},
    "wnli": {'metric': 'accuracy', 'mode': 'max'},
}


# Pre-trained models from HuggingFace zoo considered here
PRETRAINED_MODELS = [
    'bert-base-cased',
    'bert-base-uncased',
    'distilbert-base-uncased',
    'distilbert-base-cased',
    'roberta-base',
    'albert-base-v2',
    'distilroberta-base',
    'xlnet-base-cased',
    'albert-base-v1',
]


def decode_bool(hp: str) -> bool:
    # Sagemaker encodes hyperparameters in estimators as literals which are compatible with Python,
    # except for true and false that are respectively encoded as 'True' and 'False'.
    assert hp in ['True', 'False']
    return hp == 'True'


# Technical comments:
# - By default, `Tuner` is passed an experiment name for `tuner_name`, and
#   appends a date time-stamp in order to make it unique. Here, since we
#   want to use the tuner name (with postfix) also as SageMaker job name,
#   the postfix is already appended in the `tuner_name` argument. By
#   creating `Tuner` with `tuner_name_add_postfix=False`, the tuner does
#   not append a postfix.
# - We use the SageMaker checkpoint mechanism to write results (optionally
#   also logs and checkpoints) to S3. Namely, anything written to
#   `opt/ml/checkpoints` is synced to S3.
#   By default, `checkpoint_s3_uri` passed to the SM training job does not
#   depend on `tuner_name`, which leads to potentially many files from
#   other experiments being downloaded when the job starts.
#   To avoid this, we append `tuner_name` to `checkpoint_s3_uri` already.
#   We then create `Tuner` with `tuner_path_strip_name_on_sagemaker=True`,
#   to avoid the tuner name to be appended once more.
#   Note: This should probably become the default!
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='rte',
                        choices=list(TASK2METRICSMODE.keys()))
    parser.add_argument('--model_type', type=str, default='bert-base-cased',
                        choices=PRETRAINED_MODELS)
    parser.add_argument('--max_runtime', type=int, default=1800)
    parser.add_argument('--num_train_epochs', type=int, default=3,
                        help='number of epochs to train the networks')
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--optimizer', type=str, default='asha',
                        choices=('rs', 'bo', 'asha', 'mobster'))
    parser.add_argument('--tuner_name', type=str,
                        help='tuner name with date time-stamp postfix (must be unique)')
    parser.add_argument('--choose_model', type=str, default='False')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--train_valid_fraction', type=float, default=0.7)
    parser.add_argument('--store_logs_checkpoints_to_s3', type=str,
                        default='False')

    args, _ = parser.parse_known_args()
    args.choose_model = decode_bool(args.choose_model)
    args.store_logs_checkpoints_to_s3 = decode_bool(
        args.store_logs_checkpoints_to_s3)

    dataset = args.dataset
    model_type = args.model_type
    num_train_epochs = args.num_train_epochs
    seed = args.seed
    optimizer = args.optimizer

    # Path to training script. We also need to specify the names of metrics reported
    # back from this script
    entry_point = "./run_glue.py"
    metric = 'eval_' + TASK2METRICSMODE[dataset]['metric']
    mode = TASK2METRICSMODE[dataset]['mode']
    resource_attribute = 'epoch'

    # [1]
    # The configuration space contains all hyperparameters we would like to optimize,
    # and their search ranges.
    hyperparameter_space = {
        'learning_rate': loguniform(1e-6, 1e-4),
        'per_device_train_batch_size': randint(16, 48),
        'warmup_ratio': uniform(0, 0.5),
    }

    #  Additionally, it contains fixed parameters passed to the training script
    fixed_parameters = {
        'num_train_epochs': num_train_epochs,
        'model_name_or_path': model_type,
        'task_name': dataset,
        'train_valid_fraction': args.train_valid_fraction,
        'seed': seed,
        'do_train': True,
        'max_seq_length': 128,
        'output_dir': 'tmp/' + dataset,
        'evaluation_strategy': 'epoch',
        'save_strategy': 'epoch',
        'save_total_limit': 1,
    }

    config_space = {**hyperparameter_space, **fixed_parameters}

    # [2]
    # This is the default configuration provided by Hugging Face. It will always
    # be evaluated first
    default_configuration = {
        'learning_rate': 2e-5,
        'per_device_train_batch_size': 32,
        'warmup_ratio': 0.0,
    }

    # [3]
    # Combine HPO with model selection:
    # Just another categorical hyperparameter
    if args.choose_model:
        config_space['model_name_or_path'] = choice(PRETRAINED_MODELS)
        default_configuration['model_name_or_path'] = model_type

    # The backend is responsible to start and stop training evaluations. Here, we
    # use the local backend, which runs on a single instance
    backend = LocalBackend(entry_point=entry_point)

    # [4]
    # HPO algorithm
    # We can choose from these optimizers:
    schedulers = {
        'rs': RandomSearch,
        'bo': BayesianOptimization,
        'asha': ASHA,
        'mobster': MOBSTER}
    scheduler_kwargs = dict(
        metric=metric,
        mode=mode,
        random_seed=seed,  # same seed passed to training function
        points_to_evaluate=[default_configuration],  # evaluate this one first
    )
    if optimizer in {'asha', 'mobster'}:
        # The multi-fidelity methods need extra information
        scheduler_kwargs['resource_attr'] = resource_attribute
        # Maximum resource level information in `config_space`:
        scheduler_kwargs['max_resource_attr'] = 'num_train_epochs'
    scheduler = schedulers[optimizer](config_space, **scheduler_kwargs)

    # [5]
    # All parts come together in the tuner, which runs the experiment
    stop_criterion = StoppingCriterion(max_wallclock_time=args.max_runtime)
    tuner = Tuner(
        trial_backend=backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=args.n_workers,
        metadata=vars(args),  # metadata is stored along with results
        tuner_name=args.tuner_name,
        tuner_name_add_postfix=False,  # see "Technical comments"
        tuner_path_strip_name_on_sagemaker=True,  # see "Technical comments"
    )

    # Set path for logs and checkpoints
    if args.store_logs_checkpoints_to_s3:
        backend.set_path(results_root=tuner.tuner_path)
    else:
        backend.set_path(
            results_root=str(Path('~/').expanduser()), tuner_name=tuner.name)

    tuner.run()  # off we go!
