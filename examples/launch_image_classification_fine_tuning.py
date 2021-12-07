import argparse
import logging
from pathlib import Path

from syne_tune.backend.local_backend import LocalBackend
from syne_tune.optimizer.schedulers.hyperband import HyperbandScheduler, FIFOScheduler
from syne_tune.tuner import Tuner
from syne_tune.search_space import uniform, loguniform, lograndint
from syne_tune.remote.remote_launcher import RemoteLauncher
from syne_tune.stopping_criterion import StoppingCriterion
from syne_tune.optimizer.schedulers.pbt import PopulationBasedTraining
from syne_tune.optimizer.schedulers.bore_pbt import BorePopulationBasedTraining
from syne_tune.optimizer.schedulers.contextual_bo_pbt import ContextualBOPopulationBasedTraining
from syne_tune.optimizer.schedulers.contextual_blr_bo_pbt import ContextualBLRBOPopulationBasedTraining
from syne_tune.optimizer.schedulers.botorch_pbt import BOTorchPopulationBasedTraining

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--max_epoch', type=int, default=3, help='number of epochs to train the networks')
    parser.add_argument('--grace_period', type=int, default=1)
    parser.add_argument('--reduction_factor', type=int, default=3)
    parser.add_argument('--runtime', type=int, default=1800)
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--model_id', type=str, default='resnet18')
    parser.add_argument('--dataset_name', type=str, default='flower')
    parser.add_argument('--experiment_name', type=str, default='comparison')
    parser.add_argument('--search_space', type=str, default='sgd', choices=['adam', 'sgd'])
    parser.add_argument('--acquistion_function_optimization', type=str, default='gd')
    parser.add_argument('--run_local', action='store_true')
    parser.add_argument('--sagemaker_backend', action='store_true')
    parser.add_argument('--scheduler', type=str, default='pbt')
    parser.add_argument('--searcher', type=str, default='bayesopt')
    parser.add_argument('--lr_schedule', type=str, default='const')
    parser.add_argument('--population_size', type=int, default=4)
    parser.add_argument('--acquisition_function', type=str, default='lcb')
    parser.add_argument('--instance_type', type=str, default='ml.g4dn.12xlarge')
    parser.add_argument('--add_time_step_to_context', action='store_true', default=False)
    parser.add_argument('--skip_exploit', action='store_true', default=False)
    parser.add_argument('--num_opt_rounds', type=int, default=1)

    args, _ = parser.parse_known_args()

    mode = 'max'  # defines if we maximize or minimize, here we maximize accuracy
    metric = 'objective'  # the name of the metric in the report function that will be optimized
    resource_attribute = 'epoch'  # defines the resource metric in the report function
    dataset_dir = './DATA'  # specifies the local folder where the data will be stored
    entry_point = Path(__file__).parent / "training_scripts" / "fine_tuning" / "fine_tuning_image_classification.py"
    instance_type = args.instance_type

    if args.search_space == "sgd":
        config_space = {
            "epochs": args.max_epoch,
            'model_id': args.model_id,
            'dataset_dir': dataset_dir,
            'dataset_name': args.dataset_name,
            "batch_size": lograndint(4, 128),  # defines the search space for the batch size
            "learning_rate": loguniform(1e-6, 1),  # defines the search space for the learning rate
            "momentum": uniform(0.1, 0.99),
            "weight_decay": loguniform(1e-8, 1e-1),
            "optimizer_type": 'sgd'
        }
    elif args.search_space == "adam":
        config_space = {
            "epochs": args.max_epoch,
            'model_id': args.model_id,
            'dataset_dir': dataset_dir,
            'dataset_name': args.dataset_name,
            "optimizer_type": 'adam',
            "batch_size": lograndint(1, 128),  # defines the search space for the batch size
            "adam_learning_rate": loguniform(1e-4, 1),  # defines the search space for the learning rate for ADAM
        }

    # defines the backend
    backend = LocalBackend(entry_point=entry_point)

    if args.sagemaker_backend:
        from sagemaker_tune.backend.sagemaker_backend.sagemaker_backend import SagemakerBackend
        from sagemaker.pytorch import PyTorch

        role = 'arn:aws:iam::770209394645:role/service-role/AmazonSageMaker-ExecutionRole-20200813T155446'
        backend = SagemakerBackend(
            sm_estimator=PyTorch(
                source_dir=str(Path(__file__).parent / "training_scripts" / "fine_tuning"),
                entry_point=str(entry_point),
                instance_type=instance_type,
                # instance_type='local',
                instance_count=1,
                role=role,
                max_run=args.runtime,
                framework_version='1.6',
                py_version='py3',
                base_job_name=f'smt-{args.model_id}-{args.dataset_name}-{args.run_id}',
            ),
            metrics_names=[metric],
        )

    searcher = args.searcher
    search_options = {
        'num_init_random': 2
    }

    for run_id in range(args.num_runs):

        if args.scheduler in ['asha-stopping', 'asha-promotion', 'fifo']:

            config_space['scheduler_type'] = args.lr_schedule

            if args.lr_schedule == 'exp':
                config_space['gamma'] = loguniform(1e-4, 1)

            if args.scheduler == 'asha-stopping':

                scheduler = HyperbandScheduler(
                    config_space,
                    searcher=searcher,
                    type='stopping',
                    search_options=search_options,
                    max_t=args.max_epoch,
                    grace_period=args.grace_period,
                    reduction_factor=args.reduction_factor,
                    resource_attr=resource_attribute,
                    mode=mode,
                    metric=metric)

            elif args.scheduler == 'asha-promotion':

                scheduler = HyperbandScheduler(
                    config_space,
                    searcher=searcher,
                    type='promotion',
                    search_options=search_options,
                    max_t=args.max_epoch,
                    grace_period=args.grace_period,
                    reduction_factor=args.reduction_factor,
                    resource_attr=resource_attribute,
                    mode=mode,
                    metric=metric)

            elif args.scheduler == 'fifo':
                scheduler = FIFOScheduler(
                    config_space,
                    searcher=searcher,
                    mode=mode,
                    metric=metric)

        elif args.scheduler in ['pbt', 'bore-pbt', 'bo-pbt', 'blr-pbt', 'botorch-pbt']:
            config_space['scheduler_type'] = 'none'

            population_size = args.population_size

            if args.scheduler == 'pbt':
                scheduler = PopulationBasedTraining(config_space=config_space,
                                                    metric=metric,
                                                    resource_attr=resource_attribute,
                                                    population_size=population_size,
                                                    mode=mode,
                                                    max_t=args.max_epoch,
                                                    perturbation_interval=1)

            elif args.scheduler == 'bore-pbt':
                scheduler = BorePopulationBasedTraining(config_space=config_space,
                                                        metric=metric,
                                                        resource_attr=resource_attribute,
                                                        population_size=population_size,
                                                        mode=mode,
                                                        max_t=args.max_epoch,
                                                        perturbation_interval=1)

            elif args.scheduler == 'bo-pbt':
                scheduler = ContextualBOPopulationBasedTraining(config_space=config_space,
                                                                metric=metric,
                                                                resource_attr=resource_attribute,
                                                                population_size=population_size,
                                                                mode=mode,
                                                                max_t=args.max_epoch,
                                                                acquisition_function=args.acquisition_function,
                                                                perturbation_interval=1,
                                                                acquistion_function_optimization=args.acquistion_function_optimization,
                                                                do_exploit=not args.skip_exploit,
                                                                num_opt_rounds=args.num_opt_rounds,
                                                                add_time_step_to_context=args.add_time_step_to_context)
            elif args.scheduler == 'blr-pbt':
                scheduler = ContextualBLRBOPopulationBasedTraining(config_space=config_space,
                                                                   metric=metric,
                                                                   resource_attr=resource_attribute,
                                                                   population_size=population_size,
                                                                   mode=mode,
                                                                   max_t=args.max_epoch,
                                                                   acquisition_function=args.acquisition_function,
                                                                   perturbation_interval=1,
                                                                   do_exploit=not args.skip_exploit,
                                                                   num_opt_rounds=args.num_opt_rounds,
                                                                   add_time_step_to_context=args.add_time_step_to_context)
            elif args.scheduler == 'botorch-pbt':
                scheduler = BOTorchPopulationBasedTraining(config_space=config_space,
                                                                   metric=metric,
                                                                   resource_attr=resource_attribute,
                                                                   population_size=population_size,
                                                                   mode=mode,
                                                                   max_t=args.max_epoch,
                                                                   acquisition_function=args.acquisition_function,
                                                                   perturbation_interval=1,
                                                                   do_exploit=not args.skip_exploit,
                                                                   num_opt_rounds=args.num_opt_rounds,
                                                                   add_time_step_to_context=args.add_time_step_to_context)

        local_tuner = Tuner(
            backend=backend,
            scheduler=scheduler,
            max_failures=5,
            stop_criterion=StoppingCriterion(max_wallclock_time=args.runtime),
            n_workers=args.n_workers,
            tuner_name=f'smt-{args.experiment_name}-{args.dataset_name}-{args.scheduler}-{args.model_id}-{run_id}',
            metadata=vars(args)
        )

        if args.run_local:
            local_tuner.run()
        else:
            tuner = RemoteLauncher(
                tuner=local_tuner,
                instance_type=instance_type,
            )
            tuner.run(wait=False)
