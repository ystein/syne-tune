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
from syne_tune.tuner_callback import TunerCallback
from syne_tune.backend.local_backend import LocalBackend


class ComparisonSimulatedLocalBackendCallback(TunerCallback):
    """
    Callback to be used when running a tabulated blackbox via the local
    back-end. See the `nasbench201` blackbox for an example.

    If a training script for the local back-end is using the blackbox
    repository, it needs to download the blackbox data from S3, or even
    generate it. This needs to be done up front, initializing the
    training script. Once initialized, trials can be evaluated by loading
    the blackbox from its local copy. Doing the initialization as part of
    the first trial evaluation does not work. First, the download time
    would be counted as trial evaluation time, which gives skewed results for
    a comparison. Second, the local back-end starts a number of trials
    initially, one for each worker. Each of them would then try to run the
    initialization, which can even lead to file locking errors.

    For this reason, we do the initialization separately, in
    `on_tuning_start` here. This needs to be supported by the training
    script: if called with a particular config, it runs the initialization
    without doing an evaluation.

    Note: It is also important that the callback here comes before the
    :class:`StoreResultsCallback` in the list `callbacks` passed to
    :class:`Tuner`. This is because the latter sets a time stamp for the
    start of the experiment in `on_tuning_start`, and we do not want to
    have the initialization time here count towards the experiment.

    """
    def __init__(self, config_for_initialization: dict):
        """
        :param config_for_initialization: The training script is initialized
            by calling it with this particular configuration, see also
            `LocalBackend.initialize_entry_point`
        """
        self._config_for_initialization = config_for_initialization

    def on_tuning_start(self, tuner):
        backend = tuner.backend
        if isinstance(backend, LocalBackend):
            backend.initialize_entry_point(self._config_for_initialization)
