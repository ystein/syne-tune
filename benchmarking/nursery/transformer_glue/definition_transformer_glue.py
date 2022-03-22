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
"""
Fine-tuning Hugging Face transformer on GLUE sequence classification
benchmarks
"""
from pathlib import Path

from syne_tune.config_space import loguniform, randint, uniform, choice


SELECT_MODEL_ARG = 'glue_select_model'

RANDOM_SEED_ARG = 'glue_random_seed'


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

_config_space = {
    'learning_rate': loguniform(1e-6, 1e-4),
    'per_device_train_batch_size': randint(16, 48),
    'warmup_ratio': uniform(0, 0.5),
}


_fixed_parameters_defaults = {
    'model_name_or_path': 'bert-base-cased',
    'do_train': True,
    'max_seq_length': 128,
    'output_dir': 'tmp/rte',
    'evaluation_strategy': 'epoch',
    'train_valid_fraction': 0.7,
}


# This is the default configuration provided by Hugging Face.
# It should be evaluated first
default_configuration = {
    'learning_rate': 2e-5,
    'per_device_train_batch_size': 32,
    'warmup_ratio': 0.0,
}


def transformer_glue_default_params(params=None):
    if params is not None and params.get(
            'backend', 'local') == 'sagemaker':
        num_workers = 4
    else:
        num_workers = 1
    default_model = _fixed_parameters_defaults['model_name_or_path']
    select_model = params.get(SELECT_MODEL_ARG, False)
    if select_model:
        def_config = dict(default_configuration,
                          model_name_or_path=default_model)
    else:
        def_config = default_configuration
    return {
        'max_resource_level': 3,
        'dataset_name': 'rte',
        'instance_type': 'ml.g4dn.xlarge',
        'num_workers': num_workers,
        'points_to_evaluate': [def_config],
        **_fixed_parameters_defaults,
        RANDOM_SEED_ARG: None,
    }


def transformer_glue_benchmark(params):
    dataset_name = params['dataset_name']
    select_model = params.get(SELECT_MODEL_ARG, False)
    config_space = dict(
        _config_space,
        **{k: params[k] for k in _fixed_parameters_defaults.keys()},
        num_train_epochs=params['max_resource_level'],
        task_name=dataset_name,
    )
    seed = params.get(RANDOM_SEED_ARG)
    if seed is not None:
        config_space['seed'] = seed
    if select_model:
        config_space['model_name_or_path'] = choice([
            'bert-base-cased',
            'bert-base-uncased',
            'distilbert-base-uncased',
            'distilbert-base-cased',
            'roberta-base',
            'albert-base-v2',
            'distilroberta-base',
            'xlnet-base-cased',
            'albert-base-v1',
        ])
    metric = 'eval_' + TASK2METRICSMODE[dataset_name]['metric']
    mode = TASK2METRICSMODE[dataset_name]['mode']
    return {
        'script': Path(__file__).parent / 'run_glue_modified.py',
        'metric': metric,
        'mode': mode,
        'resource_attr': 'epoch',
        'max_resource_attr': 'num_train_epochs',
        'random_seed_attr': 'seed',
        'config_space': config_space,
    }
