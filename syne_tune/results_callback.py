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
from typing import Dict, Any, Optional
from time import perf_counter
import copy
import pandas as pd

from syne_tune.backend.trial_status import Trial
from syne_tune.constants import (
    ST_DECISION,
    ST_TRIAL_ID,
    ST_STATUS,
    ST_TUNER_TIME,
    ST_RESULTS_DATAFRAME_FILENAME,
    ST_FINAL_RESULTS_FILENAME,
)
from syne_tune.tuner_callback import TunerCallback
from syne_tune.util import RegularCallback, dump_json_with_numpy


class FinalResultsComposer:
    """
    Base class for ``final_results_composer`` argument in
    :class:`StoreResultsCallback`. Extracts final results at the end of
    tuning and returns them as a dictionary which can be JSON-serialized.
    This is for final results, which are not time-stamped.

    Why don't we use a lambda function instead? We would like the tuner,
    with all its dependent objects, to be dill serializable, and lambda
    functions are not.
    """

    def __call__(self, tuner) -> Optional[Dict[str, Any]]:
        """
        Called in :meth:`StoreResultsCallback.on_tuning_end`. The dictionary
        returned is stored as JSON file.

        :param tuner: :class:`~syne_tune.Tuner` object from which final results
            are extracted
        :return: Dictionary (JSON-serializable) to be written, or ``None`` if
            nothing should be written
        """
        raise NotImplementedError


class StoreResultsCallback(TunerCallback):
    """
    Default implementation of :class:`~TunerCallback` which records all
    reported results, and allows to store them as CSV file.

    :param add_wallclock_time: If ``True``, wallclock time since call of
        ``on_tuning_start`` is stored as
        :const:`~syne_tune.constants.ST_TUNER_TIME`.
    :param final_results_composer: Optional. If given, this is called in
        :meth:`on_tuning_end`, and the resulting dictionary is written as
        ``{tuner.tuner_path}/{ST_FINAL_RESULTS_FILENAME}``.
    :param store_final_results_intermediately: If ``True`` and
        ``final_results_composer``, final results are written together
        with time-stamped results, so if the experiment is terminated
        before reaching the end, the most recent results can be used.
        Defaults to ``True``.
    """

    def __init__(
        self,
        add_wallclock_time: bool = True,
        final_results_composer: Optional[FinalResultsComposer] = None,
        store_final_results_intermediately: bool = True,
    ):
        self.results = []
        self.csv_file = None
        self.save_results_at_frequency = None
        self.add_wallclock_time = add_wallclock_time
        self._final_results_composer = final_results_composer
        self._store_final_results_intermediately = store_final_results_intermediately
        self._start_time_stamp = None
        self._tuner = None

    def _set_time_fields(self, result: Dict[str, Any]):
        """
        Note that we only add wallclock time to the result if this has not
        already been done (by the backend)
        """
        if self._start_time_stamp is not None and ST_TUNER_TIME not in result:
            result[ST_TUNER_TIME] = perf_counter() - self._start_time_stamp

    def on_trial_result(
        self, trial: Trial, status: str, result: Dict[str, Any], decision: str
    ):
        assert (
            self.save_results_at_frequency is not None
        ), "on_tuning_start must always be called before on_trial_result."
        result = copy.copy(result)
        result[ST_DECISION] = decision
        result[ST_STATUS] = status
        result[ST_TRIAL_ID] = trial.trial_id

        for key in trial.config:
            result[f"config_{key}"] = trial.config[key]

        self._set_time_fields(result)

        self.results.append(result)

        if self.csv_file is not None:
            self.save_results_at_frequency()

    def store_results(self):
        """
        Store current results into CSV file, of name
        ``{tuner.tuner_path}/{ST_RESULTS_DATAFRAME_FILENAME}``.
        """
        if self.csv_file is not None:
            self.dataframe().to_csv(self.csv_file, index=False)
        if self._store_final_results_intermediately:
            self._write_final_results()

    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.results)

    def on_tuning_start(self, tuner):
        # We set the path of the csv file once the tuner is created, since the
        # path may change when the tuner is stopped and resumed again on a
        # different machine.
        self.csv_file = str(tuner.tuner_path / ST_RESULTS_DATAFRAME_FILENAME)
        # We only save results every ``results_update_frequency`` seconds as
        # this operation may be expensive on remote storage.
        self.save_results_at_frequency = RegularCallback(
            lambda: self.store_results(),
            call_seconds_frequency=tuner.results_update_interval,
        )
        if self.add_wallclock_time:
            self._start_time_stamp = perf_counter()
        if self._final_results_composer is not None:
            self._tuner = tuner

    def _write_final_results(self):
        if self._final_results_composer is not None:
            final_results = self._final_results_composer(self._tuner)
            if final_results is not None:
                path = self._tuner.tuner_path / ST_FINAL_RESULTS_FILENAME
                dump_json_with_numpy(final_results, path)

    def on_tuning_end(self):
        # Store the results in case some results were not committed yet (since
        # they are saved every ``results_update_interval`` seconds)
        self.store_results()
        if not self._store_final_results_intermediately:
            self._write_final_results()
