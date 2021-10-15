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
from sagemaker_tune.optimizer.scheduler import TrialScheduler
from sagemaker_tune.optimizer.schedulers.fifo import FIFOScheduler
from sagemaker_tune.optimizer.schedulers.hyperband import HyperbandScheduler
from sagemaker_tune.optimizer.schedulers.multiobjective.moasha import MOASHA
from sagemaker_tune.constants import SMT_WORKER_TIME
from sagemaker_tune.backend.backend import Backend
from sagemaker_tune.backend.simulator_backend.simulator_backend import SimulatorBackend

from launch_utils import make_searcher_and_scheduler
from utils import dict_get

__all__ = ['scheduler_factory',
           'short_name_scheduler_factory',
           'supported_schedulers',
           'supported_short_name_schedulers',
           'setup_scheduler_from_backend']


def _check_searcher(searcher, supported_searchers):
    assert searcher is not None, \
        "searcher needs to be provided"
    assert searcher in supported_searchers, \
        f"searcher = '{searcher}' not supported ({supported_searchers})"


supported_schedulers = {
    'fifo',
    'hyperband_stopping',
    'hyperband_promotion',
    'hyperband_cost_promotion',
    'mo_asha',
    'raytune_fifo',
    'raytune_hyperband',
}


def scheduler_factory(
        params: dict, benchmark: dict, default_params: dict) -> (
        TrialScheduler, dict):
    """
    Creates scheduler from command line parameters and benchmark descriptor.
    We also return the CL parameters extended by benchmark-specific default
    values.

    :param params: CL parameters
    :param benchmark: Benchmark descriptor
    :param default_params: Default params for benchmark
    :return: scheduler, imputed_params

    """
    params = params.copy()
    config_space = benchmark['config_space']

    scheduler = params['scheduler']
    assert scheduler in supported_schedulers, \
        f"scheduler = '{scheduler}' not supported ({supported_schedulers})"
    do_synchronous = params.get('synchronous', False)
    _default_params = dict(instance_type='ml.m4.xlarge', num_workers=4)
    _default_params.update(default_params)
    for k, v in _default_params.items():
        if params.get(k) is None:
            params[k] = v
    if params.get('searcher_num_init_random') is None:
        # The default value for this is num_workers + 2
        params['searcher_num_init_random'] = params['num_workers'] + 2

    schedulers_with_search_options = {
        'fifo',
        'hyperband_stopping',
        'hyperband_promotion',
        'hyperband_cost_promotion'}
    if scheduler in schedulers_with_search_options:
        if do_synchronous and scheduler != 'fifo':
            raise NotImplementedError(
                "Synchronous Hyperband is not currently implemented. For "
                "asynchronous Hyperband, drop the --synchronous. For "
                "synchronous FIFO, use --scheduler fifo.")
        searcher = params.get('searcher')
        if searcher is None:
            searcher = 'random'
            params['searcher'] = searcher
        else:
            supported_searchers = {'random', 'bayesopt'}
            if scheduler == 'fifo':
                supported_searchers.update(
                    {'bayesopt_cost_coarse', 'bayesopt_cost_fine',
                     'bayesopt_constrained'})
            else:
                supported_searchers.update(
                    {'bayesopt_cost', 'bayesopt_issm'})
            _check_searcher(searcher, supported_searchers)
        # Searcher and scheduler options from params
        search_options, scheduler_options = make_searcher_and_scheduler(params)
        metric = benchmark['metric']
        scheduler_options['metric'] = metric
        scheduler_options['mode'] = benchmark['mode']
        if scheduler != 'fifo':
            scheduler_options['resource_attr'] = benchmark['resource_attr']
        if scheduler == 'hyperband_cost_promotion' or searcher.startswith(
                'bayesopt_cost'):
            # Benchmark may define 'cost_attr'. If not, check for
            # 'elapsed_time_attr'
            cost_attr = None
            keys = ('cost_attr', 'elapsed_time_attr')
            for k in keys:
                if k in benchmark:
                    cost_attr = benchmark[k]
                    break
            if cost_attr is not None:
                if scheduler == 'hyperband_cost_promotion':
                    scheduler_options['cost_attr'] = cost_attr
                if searcher.startswith('bayesopt_cost'):
                    search_options['cost_attr'] = cost_attr
        k = 'points_to_evaluate'
        if k in params:
            scheduler_options[k] = params.get(k)
        k = 'map_reward'
        if k in benchmark:
            search_options[k] = benchmark[k]
        # Transfer benchmark -> search_options
        if searcher == 'bayesopt_cost_fine' or searcher == 'bayesopt_cost':
            keys = ('cost_model', 'resource_attr')
        elif searcher == 'bayesopt_constrained':
            keys = ('constraint_attr',)
        else:
            keys = ()
        for k in keys:
            v = benchmark.get(k)
            assert v is not None, \
                f"searcher = '{searcher}': Need {k} to be defined for " +\
                "benchmark"
            search_options[k] = v
        if searcher.startswith('bayesopt_cost'):
            searcher = 'bayesopt_cost'  # Internal name
        # Build scheduler and searcher
        scheduler_cls = FIFOScheduler if scheduler == 'fifo' else \
            HyperbandScheduler
        myscheduler = scheduler_cls(
            config_space,
            searcher=searcher,
            search_options=search_options,
            **scheduler_options)
    elif scheduler == 'mo_asha':
        # Use the mode for the first metric as given in the benchmark and
        # minimize time
        mode = [benchmark['mode'], 'min']
        metrics = [benchmark['metric'], SMT_WORKER_TIME]
        myscheduler = MOASHA(
            config_space,
            mode=mode,
            metrics=metrics,
            max_t=params['max_resource_level'],
            time_attr=benchmark['resource_attr'])
    else:
        from ray.tune.schedulers import AsyncHyperBandScheduler
        from ray.tune.schedulers import FIFOScheduler as RT_FIFOScheduler
        from ray.tune.suggest.skopt import SkOptSearch
        from sagemaker_tune.optimizer.schedulers.ray_scheduler import \
            RayTuneScheduler
        from sagemaker_tune.optimizer.schedulers.searchers import \
            impute_points_to_evaluate

        if do_synchronous and scheduler != 'raytune_fifo':
            raise NotImplementedError(
                "Synchronous Hyperband is not currently implemented. For "
                "asynchronous Hyperband, drop the --synchronous. For "
                "synchronous FIFO, use --scheduler raytune_fifo.")
        searcher = params.get('searcher')
        if searcher is None:
            searcher = 'random'
            params['searcher'] = searcher
        else:
            _check_searcher(searcher, {'random', 'bayesopt'})
        rt_searcher = None  # Defaults to random
        metric = benchmark['metric']
        mode = benchmark['mode']
        points_to_evaluate = impute_points_to_evaluate(
            params.get('points_to_evaluate'), config_space)
        if searcher == 'bayesopt':
            rt_searcher = SkOptSearch(points_to_evaluate=points_to_evaluate)
            points_to_evaluate = None
            rt_searcher.set_search_properties(
                mode=mode, metric=metric,
                config=RayTuneScheduler.convert_config_space(config_space))
        if scheduler == 'raytune_hyperband':
            rt_scheduler = AsyncHyperBandScheduler(
                max_t=params['max_resource_level'],
                grace_period=dict_get(params, 'grace_period', 1),
                reduction_factor=dict_get(params, 'reduction_factor', 3),
                brackets=dict_get(params, 'brackets', 1),
                time_attr=benchmark['resource_attr'],
                mode=mode,
                metric=metric)
        else:
            rt_scheduler = RT_FIFOScheduler()
            rt_scheduler.set_search_properties(metric=metric, mode=mode)
        myscheduler = RayTuneScheduler(
            config_space=config_space,
            ray_scheduler=rt_scheduler,
            ray_searcher=rt_searcher,
            points_to_evaluate=points_to_evaluate)

    return myscheduler, params


# This list is missing the hyperband_promotion types (ASHA) of
# HyperbandScheduler. It also misses that we can run model-based
# HyperbandScheduler with different surrogate models now.
_SHORT_NAMES = {
    'RS': ('fifo', 'random'),
    'GP': ('fifo', 'bayesopt'),
    'HB': ('hyperband_stopping', 'random'),  # type = promotion ?
    'Mobster': ('hyperband_stopping', 'bayesopt'),  # type = promotion ?
    'HB-raytune': ('raytune_hyperband', 'random'),
    'HB-GP-raytune': ('raytune_hyperband', 'bayesopt'),
    'MOASHA': ('mo_asha', None),
}


supported_short_name_schedulers = list(_SHORT_NAMES.keys())


def short_name_scheduler_factory(
        name: str, params: dict, benchmark: dict,
        default_params: dict) -> (TrialScheduler, dict):
    assert name in _SHORT_NAMES, \
        f"name = {name} not supported as scheduler short name " +\
        f"({list(_SHORT_NAMES.keys())})"
    scheduler, searcher = _SHORT_NAMES[name]
    params = dict(params, scheduler=scheduler, searcher=searcher)
    return scheduler_factory(params, benchmark, default_params)


def setup_scheduler_from_backend(
        scheduler: TrialScheduler, backend: Backend):
    """
    Once both scheduler and backend are created, this allows to modify the
    scheduler based on the backend.
    """
    if isinstance(backend, SimulatorBackend):
        # Those schedulers which use local timers, need to be assigned with
        # the time keeper of the simulator back-end
        if isinstance(scheduler, FIFOScheduler):
            scheduler.set_time_keeper(backend.time_keeper)