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
from typing import Dict, Any, Optional, Tuple, Callable, Union, List, Set
import itertools
import json

import pandas as pd

from syne_tune.constants import ST_METADATA_FILENAME, ST_RESULTS_DATAFRAME_FILENAME
from syne_tune.util import experiment_path


MapMetadataToSetup = Callable[[Dict[str, Any]], Optional[str]]

MapMetadataToSubplot = Callable[[Dict[str, Any]], Optional[int]]


def _strip_common_prefix(tuner_path: str) -> str:
    prefix_path = str(experiment_path())
    assert tuner_path.startswith(
        prefix_path
    ), f"tuner_path = {tuner_path}, prefix_path = {prefix_path}"
    start_pos = len(prefix_path)
    if tuner_path[start_pos] in ["/", "\\"]:
        start_pos += 1
    return tuner_path[start_pos:]


def create_index_for_result_files(
    experiment_names: Tuple[str, ...],
    metadata_to_setup: Dict[str, MapMetadataToSetup],
    metadata_to_subplot: Optional[Dict[str, MapMetadataToSubplot]] = None,
    benchmark_key: str = "benchmark",
    with_subdirs: Optional[Union[str, List[str]]] = None,
) -> (Dict[str, List[Tuple[str, str, int]]], Set[str]):
    """
    Runs over all result directories for experiments of a comparative study.
    For each experiment, we read the metadata file, extract the benchmark name
    (key ``benchmark_key``), and use ``metadata_to_setup``,
    ``metadata_to_subplot`` to map the metadata to setup name and subplot index.
    These two mappings depend on benchmark name as well. If any of the two
    return ``None``, the result is not used. Otherwise, we enter
    ``(result_path, setup_name, subplot_no)`` into the list for benchmark name.
    Here, ``result_path`` is the result path for the experiment, without the
    :meth:`~syne_tune.util.experiment_path` prefix. The index returned is the
    dictionary from benchmark names to these list. It allows to load results
    specifically for each benchmark, and we do not have to load and parse the
    metadata files again.

    If ``with_subdirs`` is given, results are loaded from subdirectories below
    each experiment name, if they match the expression ``with_subdirs``. Use
    ``subdirs="*"`` to be safe.

    :param experiment_names: Tuple of experiment names (prefixes, without the
        timestamps)
    :param metadata_to_setup: See above. Dictionary from benchmark name
    :param metadata_to_subplot: See above. Dictionary from benchmark name
    :param benchmark_key: Key for benchmark in metadata files. Defaults to
        "benchmark"
    :param with_subdirs: See above
    :return: Index, being a dictionary from benchmark name to list of
        result paths after filtering; set of setup names encountered
    """
    reverse_index = dict()
    setup_names = set()
    for experiment_name in experiment_names:
        patterns = [experiment_name + "-*/" + ST_METADATA_FILENAME]
        if with_subdirs is not None:
            if not isinstance(with_subdirs, list):
                with_subdirs = [with_subdirs]
            pattern = patterns[0]
            patterns = [experiment_name + f"/{x}/" + pattern for x in with_subdirs]
        print(f"patterns = {patterns}")
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
                    setup_name = metadata_to_setup[benchmark_name](metadata)
                except BaseException as err:
                    print(f"Caught exception for {tuner_path}:\n" + str(err))
                    raise
                if setup_name is not None:
                    if metadata_to_subplot is not None:
                        subplot_no = metadata_to_subplot[benchmark_name](metadata)
                    else:
                        subplot_no = 0
                    if subplot_no is not None:
                        if benchmark_name not in reverse_index:
                            reverse_index[benchmark_name] = []
                        reverse_index[benchmark_name].append(
                            (_strip_common_prefix(tuner_path), setup_name, subplot_no)
                        )
                        setup_names.add(setup_name)

    return reverse_index, setup_names


def load_results_dataframe_per_benchmark(
    experiment_list: List[Tuple[str, str, int]]
) -> pd.DataFrame:
    dfs = []
    for tuner_path, setup_name, subplot_no in experiment_list:
        tuner_path = experiment_path() / tuner_path
        # Process time-stamped results (results.csv.zip)
        df_filename = str(tuner_path / ST_RESULTS_DATAFRAME_FILENAME)
        try:
            df = pd.read_csv(df_filename)
        except FileNotFoundError:
            df = None
        except Exception as ex:
            print(f"{df_filename}: Error in pd.read_csv\n{ex}")
            df = None
        if df is None:
            if verbose:
                print(
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


# HIER: Anything else to extract here??


def load_results_by_experiment_names(
    experiment_names: Tuple[str, ...],
    metadata_to_setup: Callable[[dict], Optional[str]],
    metadata_print_values: Optional[List[str]] = None,
    metadata_to_subplot: Optional[Callable[[dict], Optional[int]]] = None,
    verbose: bool = False,
    with_subdirs: Optional[Union[str, List[str]]] = None,
) -> Dict[str, Any]:
    """
    Loads results into a dataframe similar to ``load_experiments_task``, but with
    some differences. First, tuner names are selected by their prefixes (in
    ``experiment_names``).

    Second, grouping and filtering is done only w.r.t. meta, so that only
    ``metadata.json`` and ``results.csv.zip`` are needed. ``metadata_to_setup``
    maps a metadata dict to a setup name or ``None.`` Runs mapped to ``None``
    are to be filtered out, while runs mapped to the same setup name are samples
    for that setup, giving rise to statistics which are plotted for that setup.
    Here, multiple setups are compared against each other in the same plot.

    ``metadata_to_subplot`` is optional. If given, it serves as grouping into
    subplots. Runs mapped to None are filtered out as well.

    If ``with_subdirs`` is given, results are loaded from subdirectories below
    each experiment name, if they match the expression ``with_subdirs``. Use
    ``subdirs="*"`` to be safe.

    :param experiment_names:
    :param metadata_to_setup:
    :param metadata_print_values: If given, list of metadata keys. Values for
        these keys are collected from metadata, and lists are printed
    :param metadata_to_subplot:
    :param with_subdirs: See above
    :return: Dictionary with keys "results_df", "setup_names"
    """
    dfs = []
    setup_names = set()
    if metadata_print_values is None:
        metadata_print_values = []
    for experiment_name in experiment_names:
        patterns = [experiment_name + "-*/metadata.json"]
        if with_subdirs is not None:
            if not isinstance(with_subdirs, list):
                with_subdirs = [with_subdirs]
            pattern = patterns[0]
            patterns = [experiment_name + f"/{x}/" + pattern for x in with_subdirs]
        print(f"patterns = {patterns}")
        metadata_values = {k: dict() for k in metadata_print_values}
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
                try:
                    setup_name = metadata_to_setup(metadata)
                except BaseException as err:
                    print(f"Caught exception for {tuner_path}:\n" + str(err))
                    raise
                if setup_name is not None:
                    if metadata_to_subplot is not None:
                        subplot_no = metadata_to_subplot(metadata)
                    else:
                        subplot_no = 0
                    if subplot_no is not None:
                        # Process time-stamped results (results.csv.zip)
                        try:
                            df = pd.read_csv(tuner_path / "results.csv.zip")
                        except FileNotFoundError:
                            df = None
                        except Exception as ex:
                            print(
                                f"{tuner_path / 'results.csv.zip'}: Error in pd.read_csv\n{ex}"
                            )
                            df = None
                        if df is None:
                            if verbose:
                                print(
                                    f"{tuner_path}: Meta-data matches filter, but "
                                    "results file not found. Skipping."
                                )
                        else:
                            df["setup_name"] = setup_name
                            setup_names.add(setup_name)
                            df["subplot_no"] = subplot_no
                            df["tuner_name"] = tuner_path.name
                            # Add all metadata fields to the dataframe
                            # for k, v in metadata.items():
                            #    if isinstance(v, list):
                            #        print(f"{k}: {v}")
                            #        for i, vi in enumerate(v):
                            #            df[k + '_%d' % i] = vi
                            #    else:
                            #        df[k] = v
                            dfs.append(df)
                            for name in metadata_print_values:
                                if name in metadata:
                                    dct = metadata_values[name]
                                    value = metadata[name]
                                    if setup_name in dct:
                                        dct[setup_name].append(value)
                                    else:
                                        dct[setup_name] = [value]

        if metadata_print_values:
            parts = []
            for name in metadata_print_values:
                dct = metadata_values[name]
                parts.append(f"{name}:")
                for setup_name, values in dct.items():
                    parts.append(f"  {setup_name}: {values}")
            print("\n".join(parts))

    if not dfs:
        res_df = None
    else:
        res_df = pd.concat(dfs, ignore_index=True)
    result = {
        "results_df": res_df,
        "setup_names": setup_names,
    }
    return result
