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
from typing import Optional, List

from syne_tune import Tuner
from syne_tune.remote.remote_launcher import RemoteLauncher
from syne_tune.backend.sagemaker_backend.sagemaker_backend import \
    SageMakerBackend


class RemoteSequentialLauncher(RemoteLauncher):
    """
    Variant of :class:`RemoteLauncher`, where a number of experiments
    are run in sequence in the SageMaker training job. This makes sense
    if each of the experiments are short, for example because the
    simulator backend is used. The main use case is to run repetitions
    of the same experiment with different random seeds.

    The different experiments are specified by a list of `Tuner` objects.
    There are some restrictions. The trial backends of all tuners must
    use the same entrypoint. The SageMaker backend is not supported.

    The name of the first tuner in the list is used as name for the
    SM training job. Results are stored in

        {self.s3_path}/{self.tuner.name}/{num}/

    where `num` is the index of the tuner in the list. The names of the
    remaining tuners are not used.

    """
    def __init__(
            self,
            tuners: List[Tuner],
            role: Optional[str] = None,
            instance_type: str = "ml.m5.xlarge",
            dependencies: Optional[List[str]] = None,
            store_logs_localbackend: bool = False,
            log_level: Optional[int] = None,
            s3_path: Optional[str] = None,
            no_tuner_logging: bool = False,
            **estimator_kwargs,
    ):
        self._assert_valid_tuners(tuners)
        # Note: `self.tuner` is the first tuner in the list
        super().__init__(
            tuner=tuners[0],
            role=role, instance_type=instance_type,
            dependencies=dependencies,
            store_logs_localbackend=store_logs_localbackend,
            log_level=log_level, s3_path=s3_path,
            no_tuner_logging=no_tuner_logging, **estimator_kwargs)
        self.tuners = tuners.copy()

    @staticmethod
    def _assert_valid_tuners(tuners):
        assert len(tuners) > 1, \
            "For a single tuner, please use RemoteLauncher"
        entrypoint_path = str(tuners[0].trial_backend.entrypoint_path())
        assert all(not isinstance(tuner.trial_backend, SageMakerBackend)
                   for tuner in tuners), \
            "SageMaker backend not supported, please use RemoteLauncher"
        assert all(str(tuner.trial_backend.entrypoint_path()) == entrypoint_path
                   for tuner in tuners[1:]), \
            "The trial backends of all tuners must use the same entrypoint path"

    def save_tuner_for_upload(self):
        # The different tuners are stored in subdirectories
        upload_path = self.upload_dir()
        for i, tuner in enumerate(self.tuners):
            tuner_path = upload_path / str(i)
            tuner_path.mkdir(exist_ok=True)
            self._save_tuner_for_upload(
                upload_dir=str(tuner_path), tuner=tuner)

    def run(self, wait: bool = True):
        # The entry point is different
        self._run(wait=wait, entry_point='remote_sequential_main.py')
