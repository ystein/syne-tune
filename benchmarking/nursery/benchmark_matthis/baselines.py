from dataclasses import dataclass
from typing import Dict, Optional

from syne_tune.optimizer.schedulers.hyperband import HyperbandScheduler
from syne_tune.optimizer.schedulers.synchronous import (
    SynchronousGeometricHyperbandScheduler,
)
from syne_tune.config_space import (
    Categorical,
    ordinal,
)


@dataclass
class MethodArguments:
    config_space: Dict
    metric: str
    mode: str
    random_seed: int
    max_resource_attr: str
    resource_attr: str
    num_brackets: Optional[int] = None
    verbose: Optional[bool] = False
    num_samples: int = 20


class Methods:
    ASHA = "ASHA"
    MOBSTER_JOINT = "MOBSTER-JOINT"
    MOBSTER_INDEP = "MOBSTER-INDEP"
    HYPERTUNE_INDEP = "HYPERTUNE-INDEP"
    SYNCHB = "SYNCHB"
    BOHB = "BOHB"
    MOBSTER_JOINT_ORD = "MOBSTER-JOINT-ORD"
    MOBSTER_INDEP_ORD = "MOBSTER-INDEP-ORD"
    HYPERTUNE_INDEP_ORD = "HYPERTUNE-INDEP-ORD"
    BOHB_ORD = "BOHB-ORD"


def _convert_categorical_to_ordinal(args: MethodArguments) -> dict:
    return {
        name: (
            ordinal(domain.categories) if isinstance(domain, Categorical) else domain
        )
        for name, domain in args.config_space.items()
    }


def _search_options(args: MethodArguments) -> dict:
    if args.verbose:
        return dict()
    else:
        return {"debug_log": False}


methods = {
    Methods.ASHA: lambda method_arguments: HyperbandScheduler(
        config_space=method_arguments.config_space,
        searcher="random",
        type="promotion",
        search_options=_search_options(method_arguments),
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_resource_attr=method_arguments.max_resource_attr,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
        brackets=method_arguments.num_brackets,
    ),
    Methods.MOBSTER_JOINT: lambda method_arguments: HyperbandScheduler(
        config_space=method_arguments.config_space,
        searcher="bayesopt",
        type="promotion",
        search_options=_search_options(method_arguments),
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_resource_attr=method_arguments.max_resource_attr,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
        brackets=method_arguments.num_brackets,
    ),
    Methods.MOBSTER_INDEP: lambda method_arguments: HyperbandScheduler(
        config_space=method_arguments.config_space,
        searcher="bayesopt",
        type="promotion",
        search_options=dict(
            _search_options(method_arguments),
            model="gp_independent",
        ),
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_resource_attr=method_arguments.max_resource_attr,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
        brackets=method_arguments.num_brackets,
    ),
    Methods.HYPERTUNE_INDEP: lambda method_arguments: HyperbandScheduler(
        config_space=method_arguments.config_space,
        searcher="bayesopt",
        type="promotion",
        search_options=dict(
            _search_options(method_arguments),
            model="gp_independent",
            hypertune_distribution_num_samples=method_arguments.num_samples,
        ),
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_resource_attr=method_arguments.max_resource_attr,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
        brackets=method_arguments.num_brackets,
    ),
    Methods.SYNCHB: lambda method_arguments: SynchronousGeometricHyperbandScheduler(
        config_space=method_arguments.config_space,
        searcher="random",
        search_options=_search_options(method_arguments),
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_resource_attr=method_arguments.max_resource_attr,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
        brackets=method_arguments.num_brackets,
    ),
    Methods.BOHB: lambda method_arguments: SynchronousGeometricHyperbandScheduler(
        config_space=method_arguments.config_space,
        searcher="kde",
        search_options=_search_options(method_arguments),
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_resource_attr=method_arguments.max_resource_attr,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
        brackets=method_arguments.num_brackets,
    ),
    Methods.MOBSTER_JOINT_ORD: lambda method_arguments: HyperbandScheduler(
        config_space=_convert_categorical_to_ordinal(method_arguments),
        searcher="bayesopt",
        type="promotion",
        search_options=_search_options(method_arguments),
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_resource_attr=method_arguments.max_resource_attr,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
        brackets=method_arguments.num_brackets,
    ),
    Methods.MOBSTER_INDEP_ORD: lambda method_arguments: HyperbandScheduler(
        config_space=_convert_categorical_to_ordinal(method_arguments),
        searcher="bayesopt",
        type="promotion",
        search_options=dict(
            _search_options(method_arguments),
            model="gp_independent",
        ),
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_resource_attr=method_arguments.max_resource_attr,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
        brackets=method_arguments.num_brackets,
    ),
    Methods.HYPERTUNE_INDEP_ORD: lambda method_arguments: HyperbandScheduler(
        config_space=_convert_categorical_to_ordinal(method_arguments),
        searcher="bayesopt",
        type="promotion",
        search_options=dict(
            _search_options(method_arguments),
            model="gp_independent",
            hypertune_distribution_num_samples=method_arguments.num_samples,
        ),
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_resource_attr=method_arguments.max_resource_attr,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
        brackets=method_arguments.num_brackets,
    ),
    Methods.BOHB_ORD: lambda method_arguments: SynchronousGeometricHyperbandScheduler(
        config_space=_convert_categorical_to_ordinal(method_arguments),
        searcher="kde",
        search_options=_search_options(method_arguments),
        mode=method_arguments.mode,
        metric=method_arguments.metric,
        max_resource_attr=method_arguments.max_resource_attr,
        resource_attr=method_arguments.resource_attr,
        random_seed=method_arguments.random_seed,
        brackets=method_arguments.num_brackets,
    ),
}
