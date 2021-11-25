import time
import logging
import numpy as np

from functools import partial
from scipy.optimize import fmin_l_bfgs_b
from scipy.stats import norm

from syne_tune.search_space import non_constant_hyperparameter_keys

from syne_tune.optimizer.schedulers.model_based_pbt import ModelBasedPopulationBasedTraining
from syne_tune.optimizer.schedulers.blr import BayesianLinearRegression, quadratic_basis_func

logger = logging.getLogger(__name__)


def drop_fixed_hyperparameters(config, config_space):
    new_config = dict()
    for name in non_constant_hyperparameter_keys(config_space):
        new_config[name] = config[name]
    return new_config


class ContextualBLRBOPopulationBasedTraining(ModelBasedPopulationBasedTraining):
    """

    """

    def _fit_model(self):

        # shape data
        X, y = [], []
        for i, row in enumerate(self.history):

            xi = []
            for k in non_constant_hyperparameter_keys(self.config_space):
                xi.append(row[k])

            xi.append(row['last_score_previous_time_step'])

            if self.add_time_step_to_context:
                xi.append(row['t'])

            X.append(xi)
            y.append(float(row['current_score']))

        print(f'num data points = {len(X)}')
        self.model = BayesianLinearRegression(basis_func=quadratic_basis_func)
        self.model.train(np.array(X), np.array(y))
        self.current_incumbent = np.min(y)

    def _select_new_candidate(self, context):
        print('start optimizer')

        print(f'current context {context}')
        st = time.time()

        def lcb(x, context):
            m, s = self.model.predict(np.concatenate((x, context))[None, :])
            return m - s

        def ei(x, context, y_star):
            m, s, dmean_dx, dstandard_deviation_dx = self.model.predict_with_grad(np.concatenate((x, context))[None, :])

            m, s = self.model.predict(np.concatenate((x, context))[None, :])

            z = (y_star - m) / s
            f = (y_star - m) * norm.cdf(z) + s * norm.pdf(z)

            df_dx = dstandard_deviation_dx * norm.pdf(z) - norm.cdf(z) * dmean_dx

            return f[0], df_dx[0, :-len(context)]

        if self.acquisition_function == 'lcb':
            acquisition = partial(lcb, context=context)
        elif self.acquisition_function == 'ei':
            acquisition = partial(ei, context=context, y_star=self.current_incumbent)

        candidates, fvals = [], []
        for i in range(self.num_opt_rounds):
            initial_candidate = self._hp_ranges.random_config(None)
            print(f'initial candidate: {initial_candidate}')

            def optimization(initial_candiate, acquisiton):
                x0 = self._hp_ranges.to_ndarray(initial_candidate)
                bounds = self._hp_ranges.get_ndarray_bounds()
                res = fmin_l_bfgs_b(acquisition, x0=x0, bounds=bounds, maxiter=1000)

                print('results optimization: ', res)

                if res[2]['task'] == b'ABNORMAL_TERMINATION_IN_LNSRCH':
                    # this condition was copied from the old GPyOpt code
                    logger.warning(
                        f"ABNORMAL_TERMINATION_IN_LNSRCH in LBFGS, "
                        "returning original candidate"
                    )
                    return initial_candiate, acquisition(x0)[0]  # returning original candidate
                else:
                    # Clip to avoid situation where result is small epsilon out of bounds
                    a_min, a_max = zip(*bounds)
                    optimized_x = np.clip(res[0], a_min, a_max)
                    # Make sure the above clipping does really just fix numerical rounding issues in LBFGS
                    # if any bigger change was made there is a bug and we want to throw an exception
                    assert np.linalg.norm(res[0] - optimized_x) < 1e-6, (res[0], optimized_x, bounds)
                    result = self._hp_ranges.from_ndarray(optimized_x.flatten())
                    return result, res[1]

            new_candidate, fval = optimization(initial_candidate, acquisition)
            print(f'new candidate: {new_candidate} with acquisition value: {fval}')

            candidates.append(new_candidate)
            fvals.append(fval)

        opt_time = time.time() - st
        logging.info(f'acquisition optimization time: {opt_time} seconds')

        b = np.argmin(fvals)
        new_candidate = candidates[b]

        logging.info(f'return candidate {new_candidate}')

        return new_candidate
