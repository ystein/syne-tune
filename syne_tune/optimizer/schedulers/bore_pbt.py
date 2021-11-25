import time
import xgboost
import logging
import pandas as pd
import numpy as np

from typing import Callable, Dict, Optional, Type

from scipy.optimize import fmin_l_bfgs_b

from syne_tune.search_space import non_constant_hyperparameter_keys, uniform, randint
from syne_tune.backend.trial_status import Trial

from syne_tune.optimizer.scheduler import SchedulerDecision, TrialSuggestion
from syne_tune.optimizer.schedulers.pbt import PopulationBasedTraining
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.hp_ranges_factory \
    import make_hyperparameter_ranges
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.meanstd_acqfunc_impl import LCBAcquisitionFunction, EIAcquisitionFunction


logger = logging.getLogger(__name__)


def drop_fixed_hyperparameters(config, config_space):
    new_config = dict()
    for name in non_constant_hyperparameter_keys(config_space):
        new_config[name] = config[name]
    return new_config



class BorePopulationBasedTraining(PopulationBasedTraining):
    """

    """

    def __init__(self,
                 config_space: Dict,
                 metric: str,
                 mode: str,
                 max_t: int,
                 gamma: float = 0.25,
                 resource_attr: str = "time_total_s",
                 population_size: int = 4,
                 num_init_random: int = 10,
                 perturbation_interval: float = 60.0,
                 resample_probability: float = 0.25,
                 quantile_fraction: float = 0.25,
                 num_iters_acquition_function: int = 500,
                 custom_explore_fn: Optional[Callable[[Dict], Dict]] = None,
                 log_config: bool = True,
                 acquisition_function: str = 'lcb',
                 add_time_step_to_context: bool = False,
                 do_exploit: bool = True,
                 num_opt_rounds: int = 1,
                 **kwargs,
                 ):

        self.model = None
        self.gamma = gamma
        self.add_time_step_to_context = add_time_step_to_context
        self.num_init_random = num_init_random
        # self.context_attr = context_attr if context_attr is not None else []
        self.num_iters_acquition_function = num_iters_acquition_function
        self.do_exploit = do_exploit
        self.num_opt_rounds = num_opt_rounds

        if acquisition_function == 'lcb':
            self.acquisition_function_class = LCBAcquisitionFunction
        elif acquisition_function == 'ei':
            self.acquisition_function_class = EIAcquisitionFunction

        self.dummy_config = {}
        for k in non_constant_hyperparameter_keys(config_space):
            self.dummy_config[k] = 0

        columns = []
        for k in non_constant_hyperparameter_keys(config_space):
            columns.append(k)
        for k in non_constant_hyperparameter_keys(config_space):
            columns.append(f'previous_{k}')

        columns.append('t')
        columns.append('reward')
        columns.append('objective')
        columns.append('previous_trial_id')
        columns.append('trial_id')

        self.history = []

        search_space = dict()
        for k in non_constant_hyperparameter_keys(config_space):
            search_space[k] = config_space[k]
        self._hp_ranges = make_hyperparameter_ranges(search_space)

        self.pending_configurations = []

        super().__init__(config_space, metric=metric, mode=mode, resource_attr=resource_attr,
                         population_size=population_size, perturbation_interval=perturbation_interval,
                         quantile_fraction=quantile_fraction,
                         resample_probability=resample_probability, custom_explore_fn=custom_explore_fn,
                         log_config=log_config, max_t=max_t, **kwargs)

    def _fit_model(self):

        # # shape data
        # evals = []
        # for i, row in enumerate(self.history):
        #
        #     # if np.isnan(row['previous_trial_id']):
        #     #     continue
        #
        #     config = dict()
        #     for k in non_constant_hyperparameter_keys(self.config_space):
        #         config[k] = row[k]
        #
        #     config['previous_objective'] = row['last_score_previous_time_step']
        #
        #     if self.add_time_step_to_context:
        #         config['t'] = row['t']
        #
        #     y = float(row['current_score'])
        #     evals.append(CandidateEvaluation(candidate=config, metrics={'y': y}))
        #
        # n_dims = len(evals[0].candidate.keys())
        #
        # if len(evals) <= self.num_init_random:
        #     return
        #
        # gp = GaussianProcessRegression(Matern52(dimension=n_dims, ARD=True))
        # print('num datapoints for GP model', len(evals))
        #
        # extended_search_space = dict()
        # for k in non_constant_hyperparameter_keys(self.config_space):
        #     extended_search_space[k] = self.config_space[k]
        #
        # all_y = [e.candidate['previous_objective'] for e in evals]
        # y_min = np.min(all_y)
        # y_max = np.max(all_y)
        # extended_search_space['previous_objective'] = uniform(y_min, y_max)
        #
        # if self.add_time_step_to_context:
        #     extended_search_space['t'] = randint(0, self._max_t)
        #
        # state = TuningJobState(hp_ranges=make_hyperparameter_ranges(extended_search_space),
        #                        candidate_evaluations=evals, failed_candidates=[], pending_evaluations=[])
        #
        # factory = GaussProcEmpiricalBayesModelFactory(active_metric='y', gpmodel=gp, num_fantasy_samples=1)
        #
        # self.model = factory.model(state, fit_params=True)


        l = []
        for k in non_constant_hyperparameter_keys(self.config_space):
            l.append(np.array([hi[k] for hi in self.history]))
        context = [hi['last_score_previous_time_step'] for hi in self.history]
        y = [hi['current_score'] for hi in self.history]

        X = np.array([*l, context]).T
        y = np.array(y)

        tau = np.quantile(y, q=self.gamma)
        labels = np.less(y, tau)

        # labels = np.zeros(Y.shape)
        # for ti in np.unique(context):
        #     idx = np.where(X[:, -1] == ti)  # find all configurations at the current time step
        #     y_ = Y[idx]
        #     tau = np.quantile(y_, q=self.gamma)  # estimate quantile
        #     z = np.less(y_, tau)
        #     labels[idx] = z  # label data points

        labels = np.array(labels, dtype=bool)

        self.model = xgboost.XGBClassifier(use_label_encoder=False)
        self.model.fit(X, labels)

    def _get_trial_id_to_continue(self, trial: Trial):
        if self.do_exploit:
            return super()._get_trial_id_to_continue(trial)
        else:
            return trial.trial_id

    def on_trial_result(self, trial: Trial, result: Dict) -> str:

        status = super().on_trial_result(trial, result)

        # bookkeeping
        config = drop_fixed_hyperparameters(trial.config, self.config_space)

        # find current config in pending list
        for index, pending in enumerate(self.pending_configurations):

            if config == pending['config']:
                get_info = pending
                self.pending_configurations.pop(index)
                break

        row = {}

        for k in config:
            row[k] = config[k]

        previous_config = get_info['previous_config']
        for k in config:
            if previous_config is None:
                row['previous_' + k] = None
            else:
                row['previous_' + k] = previous_config[k]

        if self.mode == 'min':
            y_t = result[self.metric]
        else:
            y_t = - result[self.metric]

        last_score_previous_time_step = get_info['last_score_previous_time_step']
        if last_score_previous_time_step is None:
            last_score_previous_time_step = 0

        row['last_score_previous_time_step'] = last_score_previous_time_step
        # row['relative_reward'] = (last_score_previous_config - y_t) / y_t
        row['current_score'] = y_t
        row['previous_trial_id'] = get_info['previous_trial_id']
        row['trial_id'] = trial.trial_id
        row['t'] = result[self._resource_attr]

        self.history.append(row)

        if status == SchedulerDecision.CONTINUE:
            # update pending configuration
            pending = dict()
            pending['config'] = config
            pending['previous_trial_id'] = trial.trial_id
            pending['previous_config'] = config
            pending['last_score_previous_time_step'] = y_t
            self.pending_configurations.append(pending)

        # print('save history')
        # self.history.to_csv("history.csv")

        # Re-train model
        t = time.time()
        self._fit_model()
        training_time = time.time() - t
        logging.info(f'model training time: {training_time} seconds')
        return status

    def _explore(self, trial_to_clone: Trial) -> Dict:

        if self.model is None or len(self.history) < self.num_init_random:
            # return random configuration
            return super()._explore(trial_to_clone)

        st = time.time()
        # get last entry of the trial we will continue from
        trial_id_to_continue = trial_to_clone.trial_id
        row = None
        for hi in self.history:
            if hi['trial_id'] == trial_id_to_continue:
                row = hi
        # get the last observed value of this trial id
        current_value = row['current_score']
        context = [current_value]

        if self.add_time_step_to_context:
            time_step = row['t']
            context.append(time_step)

        # select new candidate
        new_candidate = self._select_new_candidate(context=context)

        logging.info(f'time to select new candidate: {time.time() - st} seconds')

        return new_candidate

    def loss(self, x):
        if len(x.shape) < 2:
            y = - self.model.predict_proba(x[None, :])
        else:
            y = - self.model.predict_proba(x)
        # if self.classifier in ['gp', 'mlp']:
        #     return y[:, 0]
        # else:
        #     return y[:, 1]  # return probability of class 1
        return y[:, 1]

    def _select_new_candidate(self, context):
        print('start optimizer')

        print(f'current context {context}')
        st = time.time()

        values = []
        X = []
        counter = 0
        while len(values) < self.num_iters_acquition_function and counter < 10:
            xi = self._hp_ranges.random_config(None)
            if xi not in X:
                X.append(xi)
                x_ = np.concatenate((self._hp_ranges.to_ndarray(xi), np.array(context)))
                values.append(self.loss(x_)[0])
                counter = 0
            else:
                logging.warning("Re-sampled the same configuration. Retry...")
                counter += 1  # we stop sampling if after 10 retires we are not able to find a new config

        ind = np.array(values).argmin()
        new_candidate = X[ind]

        opt_time = time.time() - st
        logging.info(f'acquisition optimization time: {opt_time} seconds')

        logging.info(f'return candidate {new_candidate}')

        return new_candidate

    def _suggest(self, trial_id: int) -> Optional[TrialSuggestion]:
        ret = super()._suggest(trial_id)

        # mark config as pending
        pending = dict()
        pending['config'] = drop_fixed_hyperparameters(ret.config, self.config_space)
        pending['previous_trial_id'] = ret.checkpoint_trial_id

        if ret.checkpoint_trial_id in self._trial_state:
            pending['previous_config'] = drop_fixed_hyperparameters(
                self._trial_state[ret.checkpoint_trial_id].trial.config, self.config_space)
            pending['last_score_previous_time_step'] = self._trial_state[ret.checkpoint_trial_id].last_score
        else:
            # dummy values if this is one of the first configurations
            pending['previous_config'] = self.dummy_config
            pending['last_score_previous_time_step'] = 0

        self.pending_configurations.append(pending)

        return ret
