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
from typing import Dict, Any, Optional, Tuple, Callable, Union, List
import itertools
import json
from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd

from syne_tune.constants import (
    ST_METADATA_FILENAME,
    ST_RESULTS_DATAFRAME_FILENAME,
    ST_TUNER_TIME,
)
from syne_tune.experiments.aggregate_results import aggregate_and_errors_over_time
from syne_tune.util import experiment_path
from syne_tune.try_import import try_import_visual_message

try:
    import matplotlib.pyplot as plt
except ImportError:
    print(try_import_visual_message())

logger = logging.getLogger(__name__)


MapMetadataToSetup = Callable[[Dict[str, Any]], Optional[str]]

MapMetadataToSubplot = Callable[[Dict[str, Any]], Optional[int]]

DEFAULT_AGGREGATE_MODE = "iqm_bootstrap"


def _strip_common_prefix(tuner_path: str) -> str:
    prefix_path = str(experiment_path())
    assert tuner_path.startswith(
        prefix_path
    ), f"tuner_path = {tuner_path}, prefix_path = {prefix_path}"
    start_pos = len(prefix_path)
    if tuner_path[start_pos] in ["/", "\\"]:
        start_pos += 1
    return tuner_path[start_pos:]


def _insert_into_dict(
    metadata_values: Dict[str, Any], benchmark_name: str, key: str, value: Any
):
    if benchmark_name not in metadata_values:
        metadata_values[benchmark_name] = dict()
    inner_dict = metadata_values[benchmark_name]
    if key in inner_dict:
        inner_dict[key].append(value)
    else:
        inner_dict[key] = [value]


def create_index_for_result_files(
    experiment_names: Tuple[str, ...],
    metadata_to_setup: MapMetadataToSetup,
    metadata_to_subplot: Optional[MapMetadataToSubplot] = None,
    metadata_keys: Optional[List[str]] = None,
    benchmark_key: str = "benchmark",
    with_subdirs: Optional[Union[str, List[str]]] = None,
) -> Dict[str, Any]:
    """
    Runs over all result directories for experiments of a comparative study.
    For each experiment, we read the metadata file, extract the benchmark name
    (key ``benchmark_key``), and use ``metadata_to_setup``,
    ``metadata_to_subplot`` to map the metadata to setup name and subplot index.
    If any of the two return ``None``, the result is not used. Otherwise, we
    enter ``(result_path, setup_name, subplot_no)`` into the list for benchmark
    name.
    Here, ``result_path`` is the result path for the experiment, without the
    :meth:`~syne_tune.util.experiment_path` prefix. The index returned is the
    dictionary from benchmark names to these list. It allows loading results
    specifically for each benchmark, and we do not have to load and parse the
    metadata files again.

    If ``with_subdirs`` is given, results are loaded from subdirectories below
    each experiment name, if they match the expression ``with_subdirs``. Use
    ``subdirs="*"`` to be safe.

    If ``metadata_keys`` is given, it contains a list of keys into the
    metadata. In this case, a nested dictionary ``metadata_values`` is
    returned, where ``metadata_values[benchmark_name][key]`` contains a list
    of metadata values for this benchmark and key in ``metadata_keys``.

    :param experiment_names: Tuple of experiment names (prefixes, without the
        timestamps)
    :param metadata_to_setup: See above
    :param metadata_to_subplot: See above. Optional
    :param metadata_keys: See above. Optional
    :param benchmark_key: Key for benchmark in metadata files. Defaults to
        "benchmark"
    :param with_subdirs: See above
    :return: Dictionary; entry "index" for index (see above); entry
        "setup_names" for setup names encountered; entry "metadata_values" see
        ``metadata_keys``
    """
    reverse_index = dict()
    setup_names = set()
    metadata_values = dict()
    if metadata_keys is None:
        metadata_keys = []
    for experiment_name in experiment_names:
        patterns = [experiment_name + "-*/" + ST_METADATA_FILENAME]
        if with_subdirs is not None:
            if not isinstance(with_subdirs, list):
                with_subdirs = [with_subdirs]
            pattern = patterns[0]
            patterns = [experiment_name + f"/{x}/" + pattern for x in with_subdirs]
        logger.info(f"Patterns for result files: {patterns}")
        for meta_path in itertools.chain(
            *[experiment_path().glob(pattern) for pattern in patterns]
        ):
            tuner_path = meta_path.parent
            try:
                with open(str(meta_path), "r") as f:
                    metadata = json.load(f)
            except FileNotFoundError:
                metadata = None
            if metadata is not None:
                assert benchmark_key in metadata, (
                    f"Metadata for tuner_path = {tuner_path} does not contain "
                    f"key {benchmark_key}:\n{metadata}"
                )
                benchmark_name = metadata[benchmark_key]
                try:
                    setup_name = metadata_to_setup(metadata)
                except BaseException as err:
                    logger.error(f"Caught exception for {tuner_path}:\n" + str(err))
                    raise
                if setup_name is not None:
                    if metadata_to_subplot is not None:
                        subplot_no = metadata_to_subplot(metadata)
                    else:
                        subplot_no = 0
                    if subplot_no is not None:
                        if benchmark_name not in reverse_index:
                            reverse_index[benchmark_name] = []
                        reverse_index[benchmark_name].append(
                            (_strip_common_prefix(tuner_path), setup_name, subplot_no)
                        )
                        setup_names.add(setup_name)
                        for key in metadata_keys:
                            if key in metadata:
                                _insert_into_dict(
                                    metadata_values, benchmark_name, key, metadata[key]
                                )

    result = {
        "index": reverse_index,
        "setup_names": setup_names,
    }
    if metadata_keys:
        result["metadata_values"] = metadata_values
    return result


def load_results_dataframe_per_benchmark(
    experiment_list: List[Tuple[str, str, int]]
) -> Optional[pd.DataFrame]:
    dfs = []
    for tuner_path, setup_name, subplot_no in experiment_list:
        tuner_path = experiment_path() / tuner_path
        df_filename = str(tuner_path / ST_RESULTS_DATAFRAME_FILENAME)
        try:
            df = pd.read_csv(df_filename)
        except FileNotFoundError:
            df = None
        except Exception as ex:
            logger.error(f"{df_filename}: Error in pd.read_csv\n{ex}")
            df = None
        if df is None:
            logger.warning(
                f"{tuner_path}: Meta-data matches filter, but "
                "results file not found. Skipping."
            )
        else:
            df["setup_name"] = setup_name
            df["subplot_no"] = subplot_no
            df["tuner_name"] = tuner_path.name
            dfs.append(df)

    if not dfs:
        res_df = None
    else:
        res_df = pd.concat(dfs, ignore_index=True)
    return res_df


def _impute_with_defaults(original, default, names: List[str]) -> Dict[str, Any]:
    result = dict()
    for name in names:
        orig_val = getattr(original, name, None)
        result[name] = getattr(default, name, None) if orig_val is None else orig_val
    return result


def _check_and_set_defaults(
    params: Dict[str, Any], default_values: List[Tuple[str, Any]]
):
    for name, def_value in default_values:
        if params[name] is None:
            assert def_value is not None, f"{name} must be given"
            params[name] = def_value


@dataclass
class SubplotParameters:
    titles: List[str] = None
    title_each_figure: bool = None
    kwargs: Dict[str, Any] = None
    legend_no: List[int] = None
    xlims: List[int] = None

    def merge_defaults(
        self, default_params: "SubplotParameters"
    ) -> "SubplotParameters":
        new_params = _impute_with_defaults(
            original=self,
            default=default_params,
            names=["titles", "title_each_figure", "kwargs", "legend_no", "xlims"],
        )
        _check_and_set_defaults(new_params, [("title_each_figure", False)])
        kwargs = new_params["kwargs"]
        assert (
            kwargs is not None and "nrows" in kwargs and "ncols" in kwargs
        ), "subplots.kwargs must be given and contain 'nrows' and 'ncols' entries"
        return SubplotParameters(**new_params)


@dataclass
class ShowTrialParameters:
    setup_name: str = None
    trial_id: int = None
    new_setup_name: str = None

    def merge_defaults(
        self, default_params: "ShowTrialParameters"
    ) -> "ShowTrialParameters":
        new_params = _impute_with_defaults(
            original=self,
            default=default_params,
            names=["setup_name", "trial_id", "new_setup_name"],
        )
        default_values = [
            ("setup_name", None),
            ("new_setup_name", None),
            ("trial_id", 0),
        ]
        _check_and_set_defaults(new_params, default_values)
        return ShowTrialParameters(**new_params)


@dataclass
class PlotParameters:
    metric: str = None
    mode: str = None
    title: str = None
    xlabel: str = None
    ylabel: str = None
    xlim: Tuple[float, float] = None
    ylim: Tuple[float, float] = None
    metric_multiplier: float = None
    tick_params: Dict[str, Any] = None
    aggregate_mode: str = None
    dpi: int = None
    grid: bool = None
    subplots: SubplotParameters = None
    show_one_trial: ShowTrialParameters = None

    def merge_defaults(self, default_params: "PlotParameters") -> "PlotParameters":
        new_params = _impute_with_defaults(
            original=self,
            default=default_params,
            names=[
                "metric",
                "mode",
                "title",
                "xlabel",
                "ylabel",
                "xlim",
                "ylim",
                "metric_multiplier",
                "tick_params",
                "aggregate_mode",
                "dpi",
                "grid",
            ],
        )
        default_values = [
            ("metric", None),
            ("mode", None),
            ("title", ""),
            ("metric_multiplier", 1),
            ("aggregate_mode", DEFAULT_AGGREGATE_MODE),
            ("dpi", 200),
            ("grid", False),
        ]
        _check_and_set_defaults(new_params, default_values)
        if self.subplots is None:
            new_params["subplots"] = default_params.subplots
        elif default_params.subplots is None:
            new_params["subplots"] = self.subplots
        else:
            new_params["subplots"] = self.subplots.merge_defaults(
                default_params.subplots
            )
        if self.show_one_trial is None:
            new_params["show_one_trial"] = default_params.show_one_trial
        elif default_params.show_one_trial is None:
            new_params["show_one_trial"] = self.show_one_trial
        else:
            new_params["show_one_trial"] = self.show_one_trial.merge_defaults(
                default_params.show_one_trial
            )
        return PlotParameters(**new_params)


class ComparativeResults:
    """
    This class loads, processes, and plots results of a comparative study,
    combining several experiments for different methods, seeds, and
    benchmarks (optional). Note that an experiment corresponds to one run
    of HPO, resulting in files :const:`~syne_tune.constants.ST_METADATA_FILENAME`
    for metadata, and :const:`~syne_tune.constants.ST_RESULTS_DATAFRAME_FILENAME`
    for time-stamped results.

    There is one comparative plot per benchmark (aggregation of results
    across benchmarks are not supported here). Results are grouped by
    setup (which usually equates to method), and then summary statistics are
    shown for each setup as function of wall-clock time. The plot can also
    have several subplots, in which case results are first grouped into
    subplot number, then setup.

    Both setup name and subplot number (optional) can be configured by the
    user, as function of metadata written for each experiment. The functions
    ``metadata_to_setup`` and ``metadata_to_subplot`` (optional) can also be
    used for filtering: results of experiments for which any of them returns
    ``None``, are not used.

    If ``with_subdirs`` is given, results are loaded from subdirectories below
    each experiment name, if they match the expression ``with_subdirs``. Use
    ``subdirs="*"`` to be safe.

    If ``metadata_keys`` is given, it contains a list of keys into the
    metadata. In this case, metadata values for these keys are extracted and
    can be retrieved with :meth:`metadata_values`.

    :param experiment_names: Tuple of experiment names (prefixes, without the
        timestamps)
    :param setups: Possible values of setup names
    :param num_runs: When grouping results w.r.t. benchmark name and setup
        name, we should end up with this many experiments. They are
        random repetitions (with different seeds), over which results are
        aggregated
    :param metadata_to_setup: See above
    :param plot_params: Parameters controlling the plot. Can be overwritten
        in :meth:`plot`
    :param metadata_to_subplot: See above. Optional
    :param benchmark_key: Key for benchmark in metadata files. Defaults to
        "benchmark"
    :param with_subdirs: See above
    """

    def __init__(
        self,
        experiment_names: Tuple[str, ...],
        setups: List[str],
        num_runs: int,
        metadata_to_setup: MapMetadataToSetup,
        plot_params: Optional[PlotParameters] = None,
        metadata_to_subplot: Optional[MapMetadataToSubplot] = None,
        benchmark_key: str = "benchmark",
        with_subdirs: Optional[Union[str, List[str]]] = None,
        metadata_keys: Optional[List[str]] = None,
    ):
        result = create_index_for_result_files(
            experiment_names=experiment_names,
            metadata_to_setup=metadata_to_setup,
            metadata_to_subplot=metadata_to_subplot,
            metadata_keys=metadata_keys,
            benchmark_key=benchmark_key,
            with_subdirs=with_subdirs,
        )
        self._reverse_index = result["index"]
        assert result["setup_names"] == set(setups), (
            f"Filtered results contain setup names {result['setup_names']}, "
            f"but should contain setup names {setups}"
        )
        self._metadata_values = (
            None if metadata_keys is None else result["metadata_values"]
        )
        self.setups = setups
        self.num_runs = num_runs
        self._default_plot_params = plot_params

    def _check_benchmark_name(self, benchmark_name: Optional[str]) -> str:
        err_msg = f"benchmark_name must be one of {list(self._reverse_index.keys())}"
        if benchmark_name is None:
            assert len(self._reverse_index) == 1, err_msg
            benchmark_name = next(self._reverse_index.keys())
        else:
            assert benchmark_name in self._reverse_index, err_msg
        return benchmark_name

    def metadata_values(self, benchmark_name: Optional[str] = None) -> Dict[str, Any]:
        benchmark_name = self._check_benchmark_name(benchmark_name)
        return self._metadata_values[benchmark_name]

    @staticmethod
    def _figure_shape(plot_params: PlotParameters) -> Tuple[int, int]:
        subplots = plot_params.subplots
        if subplots is not None:
            kwargs = subplots.kwargs
            nrows = kwargs["nrows"]
            ncols = kwargs["ncols"]
        else:
            nrows = ncols = 1
        return nrows, ncols

    def _aggregrate_results(
        self, df: pd.DataFrame, plot_params: PlotParameters
    ) -> (List[List[Dict[str, np.ndarray]]], List[str]):
        subplots = plot_params.subplots
        subplot_xlims = None if subplots is None else subplots.xlims
        fig_shape = self._figure_shape(plot_params)
        num_subplots = fig_shape[0] * fig_shape[1]
        metric = plot_params.metric
        mode = plot_params.mode
        metric_multiplier = plot_params.metric_multiplier
        xlim = plot_params.xlim
        aggregate_mode = plot_params.aggregate_mode
        show_one_trial = plot_params.show_one_trial
        do_show_one_trial = show_one_trial is not None
        setup_names = self.setups
        if do_show_one_trial:
            # Put extra name at the end
            setup_names = setup_names + [show_one_trial.new_setup_name]
        stats = [[None] * len(setup_names) for _ in range(num_subplots)]
        for (subplot_no, setup_name), setup_df in df.groupby(
            ["subplot_no", "setup_name"]
        ):
            if subplot_xlims is not None:
                xlim = (0, subplot_xlims[subplot_no])
            if do_show_one_trial and show_one_trial.setup_name == setup_name:
                num_iter = 2
            else:
                num_iter = 1
            max_rt = None
            # If this setup is named as ``show_one_trial.setup_name``, we need
            # to go over the data 2x. The first iteration is as usual, the
            # second extracts the information for the single trial and extends
            # the curve.
            for it in range(num_iter):
                one_trial_special = it == 1
                if one_trial_special:
                    # Filter down the dataframe
                    trial_id = show_one_trial.trial_id
                    setup_df = setup_df[setup_df["trial_id"] == trial_id]
                    new_setup_name = show_one_trial.new_setup_name
                    prev_max_rt = max_rt
                else:
                    new_setup_name = setup_name
                traj = []
                runtime = []
                trial_nums = []
                tuner_names = []
                max_rt = []
                for tuner_name, sub_df in setup_df.groupby("tuner_name"):
                    tuner_names.append(tuner_name)
                    if mode == "max":
                        ys = 1 - metric_multiplier * np.array(sub_df[metric].cummax())
                    else:
                        ys = metric_multiplier * np.array(sub_df[metric].cummin())
                    rt = np.array(sub_df[ST_TUNER_TIME])
                    if one_trial_special:
                        # Hack to extend curve to the end, so it can be
                        # seen
                        pos = len(runtime)
                        rt = np.append(rt, prev_max_rt[pos])
                        ys = np.append(ys, ys[-1])
                    max_rt.append(rt[-1])
                    if xlim is not None:
                        # Slice w.r.t. time. Doing this here, speeds up
                        # aggregation
                        ind = np.logical_and(rt >= xlim[0], rt <= xlim[1])
                        rt = rt[ind]
                        ys = ys[ind]
                    traj.append(ys)
                    runtime.append(rt)
                    trial_nums.append(len(sub_df.trial_id.unique()))

                setup_id = setup_names.index(setup_name)
                stats[subplot_no][setup_id] = aggregate_and_errors_over_time(
                    errors=traj, runtimes=runtime, mode=aggregate_mode
                )
                if not one_trial_special:
                    if subplots is not None:
                        msg = f"[{subplot_no}, {setup_name}]: "
                    else:
                        msg = f"[{setup_name}]: "
                    msg += f"max_rt = {np.mean(max_rt):.2f} (+- {np.std(max_rt):.2f})"
                    logger.info(msg)
                    num_repeats = len(tuner_names)
                    if num_repeats != self.num_runs:
                        if subplots is not None:
                            part = f"subplot = {subplot_no}, "
                        else:
                            part = ""
                        logger.warning(
                            f"{part}setup = {setup_name} has {num_repeats} repeats "
                            f"instead of {self.num_runs}:\n{tuner_names}"
                        )
        return stats, setup_names

    def _plot_figure(
        self, stats: List[List[Dict[str, np.ndarray]]], plot_params: PlotParameters
    ):
        subplots = plot_params.subplots
        if subplots is not None:
            subplot_xlims = subplots.xlims
            subplots_kwargs = subplots.kwargs
            ncols = subplots_kwargs["ncols"]
            nrows = subplots.kwargs["nrows"]
            subplot_titles = subplots.titles
            legend_no = [] if subplots.legend_no is None else subplots.legend_no
            if not isinstance(legend_no, list):
                legend_no = [legend_no]
            title_each_figure = subplots.title_each_figure
        else:
            subplot_xlims = None
            subplots_kwargs = dict(nrows=1, ncols=1)
            nrows = ncols = 1
            subplot_titles = None
            legend_no = [0]
            title_each_figure = False
        if subplot_titles is None:
            subplot_titles = [plot_params.title] + ["" * (ncols - 1)]
        ylim = plot_params.ylim
        xlim = plot_params.xlim  # Can be overwritten by ``subplot_xlims``
        xlabel = plot_params.xlabel
        ylabel = plot_params.ylabel
        tick_params = plot_params.tick_params

        plt.figure(dpi=plot_params.dpi)
        figsize = (5 * ncols, 4 * nrows)
        fig, axs = plt.subplots(**subplots_kwargs, squeeze=False, figsize=figsize)
        for subplot_no, stats_subplot in enumerate(stats):
            row = subplot_no % nrows
            col = subplot_no // nrows
            ax = axs[row, col]
            # Plot curves in the order of ``setups``. Not all setups may feature in
            # each of the subplots
            for i, (curves, setup_name) in enumerate(zip(stats_subplot, self.setups)):
                if curves is not None:
                    color = f"C{i}"
                    x = curves["time"]
                    ax.plot(x, curves["aggregate"], color=color, label=setup_name)
                    ax.plot(x, curves["lower"], color=color, alpha=0.4, linestyle="--")
                    ax.plot(x, curves["upper"], color=color, alpha=0.4, linestyle="--")
            if subplot_xlims is not None:
                xlim = (0, subplot_xlims[subplot_no])
            if xlim is not None:
                ax.set_xlim(*xlim)
            if ylim is not None:
                ax.set_ylim(*ylim)
            if xlabel is not None and row == nrows - 1:
                ax.set_xlabel(xlabel)
            if ylabel is not None and col == 0:
                ax.set_ylabel(ylabel)
            if tick_params is not None:
                ax.tick_params(**tick_params)
            if subplot_titles is not None:
                if title_each_figure:
                    ax.set_title(subplot_titles[subplot_no])
                elif row == 0:
                    ax.set_title(subplot_titles[col])
            if plot_params.grid:
                ax.grid(True)
            if subplot_no in legend_no:
                ax.legend()
        plt.show()
        return fig, axs

    def plot(
        self,
        benchmark_name: Optional[str] = None,
        plot_params: Optional[PlotParameters] = None,
        file_name: Optional[str] = None,
    ):
        """
        Create comparative plot from results of all experiments collected at
        construction, for benchmark ``benchmark_name`` (if there is a single
        benchmark only, this need not be given).

        If ``plot_params.show_one_trial`` is given, the metric value for a
        particular trial ``plot_params.show_one_trial.trial_id`` in a particular
        setup ``plot_params.show_one_trial.setup_name``is shown in all subplots
        the setup is contained in. This is useful to contrast the performance
        of methods against the performance for one particular trial, for example
        the initial configuration (i.e., to show how much this can be improved
        upon). The final metric value of this trial is extended until the end
        of the horizontal range, in order to make it visible. The corresponding
        curve is labeled with ``plot_params.show_one_trial.new_setup_name`` in
        the legend.

        :param benchmark_name: Name of benchmark for which to plot results.
            Not needed if there is only one benchmark
        :param plot_params: Parameters controlling the plot. Values provided
            here overwrite values provided at construction.
        :param file_name: If given, the figure is stored in a file of this name
        """
        benchmark_name = self._check_benchmark_name(benchmark_name)
        if plot_params is None:
            plot_params = PlotParameters()
        plot_params = plot_params.merge_defaults(self._default_plot_params)
        logger.info(f"Load results for benchmark {benchmark_name}")
        results_df = load_results_dataframe_per_benchmark(
            self._reverse_index[benchmark_name]
        )
        logger.info("Aggregate results")
        stats = self._aggregrate_results(results_df, plot_params)
        fig, axs = self._plot_figure(stats, plot_params)
        if file_name is not None:
            fig.savefig(file_name, dpi=plot_params.dpi)
