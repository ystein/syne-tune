import logging
import argparse

from syne_tune.backend.local_backend import LocalBackend
from syne_tune.tuner import Tuner
from syne_tune.config_space import uniform, loguniform, choice, randint
from syne_tune.stopping_criterion import StoppingCriterion
from syne_tune.optimizer.baselines import ASHA, MOBSTER, BayesianOptimization, \
    RandomSearch
from syne_tune.optimizer.schedulers.searchers.searcher_callback import \
    StoreResultsAndModelParamsCallback


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

    args, _ = parser.parse_known_args()
    args.choose_model = (args.choose_model.upper() == 'TRUE')

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

    # The configuration space contains all hyperparameters we would like to optimize,
    # and their search ranges. Additionally, it contains fixed parameters passed to
    # the training script
    hyperparameter_space = {
        'learning_rate': loguniform(1e-6, 1e-4),
        'per_device_train_batch_size': randint(16, 48),
        'warmup_ratio': uniform(0, 0.5),
    }

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
        'load_best_model_at_end': True,
        'save_strategy': 'epoch',
        'save_total_limit': 1,
    }

    config_space = {**hyperparameter_space, **fixed_parameters}

    # This is the default configuration provided by Hugging Face. It will always
    # be evaluated first
    default_configuration = {
        'learning_rate': 2e-5,
        'per_device_train_batch_size': 32,
        'warmup_ratio': 0.0,
    }

    if args.choose_model:
        # In this variant, we also choose the pre-trained model
        config_space['model_name_or_path'] = choice(PRETRAINED_MODELS)
        default_configuration['model_name_or_path'] = model_type

    # The backend is responsible to start and stop training evaluations. Here, we
    # use the local backend, which runs on a single instance
    backend = LocalBackend(entry_point=entry_point)

    # HPO algorithm
    # We use the same random seed as passed to the training function. And
    # `default_configuration` is the first config to be evaluated
    scheduler_kwargs = dict(
        metric=metric,
        mode=mode,
        random_seed=seed,
        points_to_evaluate=[default_configuration])
    if optimizer in {'asha', 'mobster'}:
        # The multi-fidelity methods need extra information
        scheduler_kwargs['resource_attr'] = resource_attribute
        # Maximum resource level information in `config_space`:
        scheduler_kwargs['max_resource_attr'] = 'num_train_epochs'
    schedulers = {
        'rs': RandomSearch,
        'bo': BayesianOptimization,
        'asha': ASHA,
        'mobster': MOBSTER}
    scheduler = schedulers[optimizer](config_space, **scheduler_kwargs)

    # All parts come together in the tuner, which runs the experiment
    # Note `args.tuner_name` already has the date time-stamp postfix (so that
    # SageMaker job name coincides with where results are stored), so `Tuner`
    # must not append it again
    stop_criterion = StoppingCriterion(max_wallclock_time=args.max_runtime)
    callbacks = [StoreResultsAndModelParamsCallback()]
    tuner = Tuner(
        trial_backend=backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=args.n_workers,
        metadata=vars(args),
        callbacks=callbacks,
        tuner_name=args.tuner_name,
        tuner_name_add_postfix=False,  # do not append date time-stamp again
    )

    tuner.run()
