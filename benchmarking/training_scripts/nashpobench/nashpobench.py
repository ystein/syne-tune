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
import os
import argparse
import logging
import time

from syne_tune.report import Reporter

from benchmarking.utils.checkpoint import resume_from_checkpointed_model, \
    checkpoint_model_at_rung_level, add_checkpointing_to_argparse
from benchmarking.utils.parse_bool import parse_bool
from benchmarking.blackbox_repository.conversion_scripts.scripts.fcnet_import import \
    METRIC_VALID_LOSS, METRIC_ELAPSED_TIME, RESOURCE_ATTR,  BLACKBOX_NAME

from syne_tune.search_space import choice, randint, uniform, loguniform

CONFIG_SPACE = {
    "hp_activation_fn_1": choice(["tanh", "relu"]),
    "hp_activation_fn_2": choice(["tanh", "relu"]),
    "hp_batch_size": randint(0, 3),
    "hp_dropout_1": randint(0, 2),
    "hp_dropout_2": randint(0, 2),
    "hp_init_lr": randint(0, 5),
    'hp_lr_schedule': choice(["cosine", "const"]),
    # 'hp_n_units_1': choice([16, 32, 64, 128, 256, 512]),
    # 'hp_n_units_2': choice([16, 32, 64, 128, 256, 512]),
    'hp_n_units_1': randint(0, 5),
    'hp_n_units_2': randint(0, 5),
}

original_config_space = {
    "hp_activation_fn_1": choice(["tanh", "relu"]),
    "hp_activation_fn_2": choice(["tanh", "relu"]),
    "hp_batch_size": choice([8, 16, 32, 64]),
    "hp_dropout_1": choice([0.0, 0.3, 0.6]),
    "hp_dropout_2": choice([0.0, 0.3, 0.6]),
    "hp_init_lr": choice([0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]),
    'hp_lr_schedule': choice(["cosine", "const"]),
    'hp_n_units_1': choice([16, 32, 64, 128, 256, 512]),
    'hp_n_units_2': choice([16, 32, 64, 128, 256, 512]),
}


def objective(config):
    dont_sleep = parse_bool(config['dont_sleep'])

    ts_start = time.time()
    s3_root = config.get('blackbox_repo_s3_root')
    blackbox = load_blackbox(
        BLACKBOX_NAME, s3_root=s3_root)[config['dataset_name']]

    essential_config = {}
    for k in CONFIG_SPACE:
        if k in ['hp_batch_size', 'hp_dropout_1', 'hp_dropout_2', 'hp_init_lr', 'hp_n_units_1', 'hp_n_units_2']:
            essential_config[k] = original_config_space[k].categories[int(config[k])]
        else:
            essential_config[k] = config[k]

    fidelity_range = (1, config['epochs'])
    all_metrics = metrics_for_configuration(
        blackbox=blackbox,
        config=essential_config,
        resource_attr=RESOURCE_ATTR,
        fidelity_range=fidelity_range)
    startup_overhead = time.time() - ts_start

    report = Reporter()

    # Checkpointing
    # Since this is a tabular benchmark, checkpointing is not really needed.
    # Still, we use a "checkpoint" file in order to store the epoch at which
    # the evaluation was paused, since this information is not passed

    def load_model_fn(local_path: str) -> int:
        local_filename = os.path.join(local_path, 'checkpoint.json')
        try:
            with open(local_filename, 'r') as f:
                data = json.load(f)
                resume_from = int(data['epoch'])
        except Exception:
            resume_from = 0
        return resume_from

    def save_model_fn(local_path: str, epoch: int):
        os.makedirs(local_path, exist_ok=True)
        local_filename = os.path.join(local_path, 'checkpoint.json')
        with open(local_filename, 'w') as f:
            json.dump({'epoch': str(epoch)}, f)

    resume_from = resume_from_checkpointed_model(config, load_model_fn)

    # Loop over epochs
    current_time = 0
    for epoch in range(resume_from + 1, config['epochs'] + 1):
        metrics_this_epoch = all_metrics[epoch - 1]
        elapsed_time = metrics_this_epoch[METRIC_ELAPSED_TIME]

        valid_error = metrics_this_epoch[METRIC_VALID_LOSS]
        time_this_epoch = elapsed_time - current_time

        current_time = elapsed_time

        if not dont_sleep:
            if epoch == resume_from + 1:
                # Subtract startup overhead of loading the table
                time_this_epoch = max(time_this_epoch - startup_overhead, 0.0)
            time.sleep(time_this_epoch)

        report_dict = {
            RESOURCE_ATTR: epoch,
            METRIC_VALID_LOSS: valid_error,
            METRIC_ELAPSED_TIME: elapsed_time}
        report(**report_dict)

        # Write checkpoint (optional)
        if (not dont_sleep) or epoch == config['epochs']:
            checkpoint_model_at_rung_level(config, save_model_fn, epoch)


if __name__ == '__main__':
    # Benchmark-specific imports are done here, in order to avoid import
    # errors if the dependencies are not installed (such errors should happen
    # only when the code is really called)
    import json

    from blackbox_repository import load as load_blackbox
    from blackbox_repository.utils import metrics_for_configuration

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--dont_sleep', type=str, required=True)
    parser.add_argument('--blackbox_repo_s3_root', type=str)
    for name in CONFIG_SPACE:
        parser.add_argument(f"--{name}", type=str, required=True)
    add_checkpointing_to_argparse(parser)

    args, _ = parser.parse_known_args()

    objective(config=vars(args))
