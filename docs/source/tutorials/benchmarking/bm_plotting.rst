Visualization of Results
========================

As we have seen, Syne Tune is a powerful tool for running a large number of
experiments in parallel, which can be used to compare different tuning
algorithms, or to split a difficult tuning problem into smaller pieces, which
can be worked on in parallel. In this section, we show results of all
experiments of such a comparative study can be visualized, using plotting
facilities provided in Syne Tune.

.. note::
   This section offers an example of the plotting facilities in Syne Tune. A
   more comprehensive tutorial is forthcoming.

A Comparative Study
-------------------

For the purpose of this tutorial, we ran the setup of
`benchmarking/nursery/benchmark_hypertune/ <../../benchmarking/benchmark_hypertune.html>`__,
using 15 random repetitions (or seeds). This is the command:

.. code-block:: bash

   python benchmarking/nursery/benchmark_hypertune/launch_remote.py \
     --experiment_tag docs-1 --random_seed 2965402734 --num_seeds 15

Note that we fix the seed here in order to obtain repeatable results. Recall
from `here <bm_simulator.html#defining-the-experiment>`__ that we compare 7
methods on 12 surrogate benchmarks:

* Since 4 of the 7 methods are "expensive", the above command launches
  ``3 + 4 * 15 = 63`` remote tuning jobs in parallel. Each of these jobs runs
  experiments for one method and all 12 benchmarks. For the "expensive" methods,
  each job runs a single seed, while for the remaining methods (ASHA, SYNCHB,
  BOHB), all seeds are run sequentially in a single job, so that a job for a
  "cheap" method runs ``12 * 15 = 180`` experiments sequentially.
* The total number of experiment runs is ``7 * 12 * 15 = 1260``
* Results of these experiments are stored to S3, using paths such as
  ``<s3-root>/syne-tune/docs-1/ASHA/docs-1-<datetime>/`` for ASHA (all seeds),
  or ``<s3-root>/syne-tune/docs-1/HYPERTUNE-INDEP-5/docs-1-<datetime>/`` for
  seed 5 of HYPERTUNE-INDEP. Result files are ``metadata.json``,
  ``results.csv.gz``, and ``tuner.dill``. The former two are required for plotting
  results.

Once all of this has finished, we are left with 3780 result files on S3. We will
now show how these can be downloaded, processed, and visualized.

Visualization of Results
------------------------

First, we need to download the results from S3 to the local disk. This can be
done by a command which is also printed at the end of ``launch_remote.py``:

.. code-block:: bash

   aws s3 sync s3://<BUCKET-NAME>/syne-tune/docs-1/ ~/syne-tune/docs-1/ \
     --exclude "*" --include "*metadata.json" --include "*results.csv.zip"

This command can also be run from inside the plotting code.

