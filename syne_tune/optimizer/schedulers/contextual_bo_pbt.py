import time
import logging
import numpy as np

from typing import Optional, Type

from scipy.optimize import fmin_l_bfgs_b

from syne_tune.search_space import non_constant_hyperparameter_keys, uniform, randint

from syne_tune.optimizer.schedulers.model_based_pbt import ModelBasedPopulationBasedTraining
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.hp_ranges_factory \
    import make_hyperparameter_ranges
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common import CandidateEvaluation
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.gp_model import GaussProcEmpiricalBayesModelFactory
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.tuning_job_state import TuningJobState
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.gp_regression import \
    GaussianProcessRegression
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.kernel import Matern52
from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.base_classes \
    import AcquisitionFunction, LocalOptimizer, SurrogateOutputModel
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.meanstd_acqfunc_impl import LCBAcquisitionFunction, EIAcquisitionFunction
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.hp_ranges \
    import HyperparameterRanges
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common \
    import Configuration

logger = logging.getLogger(__name__)


def drop_fixed_hyperparameters(config, config_space):
    new_config = dict()
    for name in non_constant_hyperparameter_keys(config_space):
        new_config[name] = config[name]
    return new_config


class LBFGSOptimizeAcquisitionContext(LocalOptimizer):
    def __init__(
            self, hp_ranges: HyperparameterRanges, model: SurrogateOutputModel, context: [],
            acquisition_function_class: Type[AcquisitionFunction], active_metric: str = None):
        super().__init__(hp_ranges, model, acquisition_function_class, active_metric)
        # Number criterion evaluations in last recent optimize call
        self.num_evaluations = None
        self.context = context

    def optimize(self, candidate: Configuration, params_acquisition_function:dict = {},
                 model: Optional[SurrogateOutputModel] = None) -> Configuration:
        # Before local minimization, the model for this state_id should have been fitted.
        if model is None:
            model = self.model
        acquisition_function = self.acquisition_class(model, active_metric=self.active_metric, **params_acquisition_function)

        x0 = self.hp_ranges.to_ndarray(candidate)
        bounds = self.hp_ranges.get_ndarray_bounds()
        n_evaluations = [0]  # wrapped in list to allow access from function

        # unwrap 2d arrays
        def f_df(x):
            n_evaluations[0] += 1
            x_ = np.concatenate((x, np.array(self.context)), axis=0)

            a, g = acquisition_function.compute_acq_with_gradient(x_)
            return a, g[:-len(self.context)]

        res = fmin_l_bfgs_b(f_df, x0=x0, bounds=bounds, maxiter=1000, disp=10)
        self.num_evaluations = n_evaluations[0]
        if res[2]['task'] == b'ABNORMAL_TERMINATION_IN_LNSRCH':
            # this condition was copied from the old GPyOpt code
            logger.warning(
                f"ABNORMAL_TERMINATION_IN_LNSRCH in lbfgs after {n_evaluations[0]} evaluations, "
                "returning original candidate"
            )
            return candidate, acquisition_function.compute_acq(candidate)  # returning original candidate
        else:
            # Clip to avoid situation where result is small epsilon out of bounds
            a_min, a_max = zip(*bounds)
            optimized_x = np.clip(res[0], a_min, a_max)
            # Make sure the above clipping does really just fix numerical rounding issues in LBFGS
            # if any bigger change was made there is a bug and we want to throw an exception
            assert np.linalg.norm(res[0] - optimized_x) < 1e-6, (res[0], optimized_x, bounds)
            result = self.hp_ranges.from_ndarray(optimized_x.flatten())
            return result, res[1]


class ContextualBOPopulationBasedTraining(ModelBasedPopulationBasedTraining):
    """

    """
    def _fit_model(self):

        # shape data
        evals = []
        for i, row in enumerate(self.history):

            # if np.isnan(row['previous_trial_id']):
            #     continue

            config = dict()
            for k in non_constant_hyperparameter_keys(self.config_space):
                config[k] = row[k]

            config['previous_objective'] = row['last_score_previous_time_step']

            if self.add_time_step_to_context:
                config['t'] = row['t']

            y = float(row['current_score'])
            evals.append(CandidateEvaluation(candidate=config, metrics={'y': y}))

        n_dims = len(evals[0].candidate.keys())

        if len(evals) <= self.num_init_random:
            return

        gp = GaussianProcessRegression(Matern52(dimension=n_dims, ARD=True))
        print('num datapoints for GP model', len(evals))

        extended_search_space = dict()
        for k in non_constant_hyperparameter_keys(self.config_space):
            extended_search_space[k] = self.config_space[k]

        all_y = [e.candidate['previous_objective'] for e in evals]
        y_min = np.min(all_y)
        y_max = np.max(all_y)
        extended_search_space['previous_objective'] = uniform(y_min, y_max)

        if self.add_time_step_to_context:
            extended_search_space['t'] = randint(0, self._max_t)

        state = TuningJobState(hp_ranges=make_hyperparameter_ranges(extended_search_space),
                               candidate_evaluations=evals, failed_candidates=[], pending_evaluations=[])

        factory = GaussProcEmpiricalBayesModelFactory(active_metric='y', gpmodel=gp, num_fantasy_samples=1)

        self.model = factory.model(state, fit_params=True)

    def _select_new_candidate(self, context):
        print('start optimizer')

        params_acquisition_function = dict()
        if self.acquisition_function == 'lcb':
            acquisition_function_class = LCBAcquisitionFunction
            params_acquisition_function['kappa'] = 1.0
        elif self.acquisition_function == 'ei':
            acquisition_function_class = EIAcquisitionFunction

        print(f'current context {context}')
        st = time.time()

        opt = LBFGSOptimizeAcquisitionContext(hp_ranges=self._hp_ranges,
                                              acquisition_function_class=acquisition_function_class,
                                              model=self.model, context=context)

        candidates, fvals = [], []
        for i in range(self.num_opt_rounds):
            initial_candidate = self._hp_ranges.random_config(None)
            print(f'initial candidate: {initial_candidate}')
            new_candidate, fval = opt.optimize(candidate=initial_candidate, params_acquisition_function=params_acquisition_function)
            print(f'new candidate: {new_candidate} with acquisition value: {fval}')

            candidates.append(new_candidate)
            fvals.append(fval)

        opt_time = time.time() - st
        logging.info(f'acquisition optimization time: {opt_time} seconds')

        b = np.argmin(fvals)
        new_candidate = candidates[b]

        logging.info(f'return candidate {new_candidate}')

        return new_candidate
