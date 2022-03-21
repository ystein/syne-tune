from pathlib import Path

from syne_tune.backend.local_backend import LocalBackend
from syne_tune.tuner import Tuner
from syne_tune.config_space import uniform, loguniform, randint
from syne_tune.stopping_criterion import StoppingCriterion
from syne_tune.optimizer.baselines import ASHA
from syne_tune.remote.remote_launcher import RemoteLauncher


# For different GLUE tasks: Metric to optimize (prefix 'eval_'), and whether
# to maximize or minimize
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


dataset = 'rte'  # GLUE task
model_type = 'bert-base-cased'  # pre-trained model

experiment_name = 'hpo-glue-basic-' + dataset
max_runtime = 1800  # Run HPO for this many seconds
num_train_epochs = 3  # Maximum number of training epochs
# instance_type = 'ml.g4dn.xlarge'  # AWS instance type
# n_workers = 1  # Number of HPO evaluations in parallel (needs sufficient GPUs)
instance_type = 'ml.g4dn.12xlarge'  # AWS instance type
n_workers = 4  # Number of HPO evaluations in parallel (needs sufficient GPUs)
seed = 1234


# Path to training script. We also need to specify the names of metrics reported
# back from this script
entry_point = "./run_glue.py"

metric = 'eval_' + TASK2METRICSMODE[dataset]['metric']
mode = TASK2METRICSMODE[dataset]['mode']
resource_attr = 'epoch'


# The configuration space contains all hyperparameters we would like to optimize,
# and their search ranges. Additionally, it contains fixed parameters passed to
# the training script

hyperparameter_space = {
    'learning_rate': loguniform(1e-6, 1e-4),
    'per_device_train_batch_size': randint(16, 48),
    'warmup_ratio': uniform(0, 0.5),
}

# This is the default configuration provided by Hugging Face. This will always
# be evaluated first
default_configuration = {
    'learning_rate': 2e-5,
    'per_device_train_batch_size': 32,
    'warmup_ratio': 0.0,
}

fixed_parameters = {
    'num_train_epochs': num_train_epochs,
    'model_name_or_path': model_type,
    'task_name': dataset,
    'do_train': True,
    'max_seq_length': 128,
    'seed': seed,
    'output_dir': 'tmp/' + dataset,
    'evaluation_strategy': 'epoch',
}

config_space = {**hyperparameter_space, **fixed_parameters}


# The backend is responsible to start and stop training evaluations. Here, we
# use the local backend, which runs on a single instance. Later examples will
# showcase a more powerful backend
backend = LocalBackend(entry_point=entry_point)


# We use ASHA (asynchronous successive halving) as HPO method, which samples
# configurations randomly and terminates poorly performing configurations
# early (e.g., after one epoch).
# By passing `default_configuration` for `points_to_evaluate`, we ensure this
# is the first configuration to be evaluated (it will not be stopped early).
scheduler = ASHA(
    config_space,
    metric=metric,
    mode=mode,
    resource_attr=resource_attr,
    max_t=num_train_epochs,
    random_seed=seed,
    points_to_evalute=[default_configuration],
)


# All parts come together in the tuner, which runs the experiment
stop_criterion = StoppingCriterion(max_wallclock_time=max_runtime)
local_tuner = Tuner(
    trial_backend=backend,
    scheduler=scheduler,
    stop_criterion=stop_criterion,
    n_workers=n_workers,
    tuner_name=experiment_name,
)

#local_tuner.run()  # would run experiment locally


# Instead of running the experiment locally, we'll launch a SageMaker training
# job. This allows us to launch from anywhere without having to worry about
# GPUs or dependencies being installed.

# `RemoteLauncher` is one of the ways Syne Tune offers to run experiments
# remotely. It requires some setup work and allows to use a single SM
# estimator type (`PyTorch`), but is easy to use, since only a single
# launch script is required.
estimator_kwargs = {
    'disable_profiler': True,
    'max_run': int(1.01 * max_runtime),
}
remote_tuner = RemoteLauncher(
    tuner=local_tuner,
    instance_type=instance_type,
    **estimator_kwargs,
)

remote_tuner.run(wait=False)
