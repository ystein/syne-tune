Distributed Tuning with SageMaker
=================================

As models to be tuned become larger, the *local backend* becomes less attractive.
It can only distribute training jobs on a single instance, which is limited by
the number of GPUs offered there. For larger workloads, the *SageMaker backend*
should be preferred, as we demonstrate here.

Scripts for Experiments
-----------------------

For this example, we consider a larger NLP transformer model trained on the
WikiText-2 corpus. The training script is
`training_script.py <../../training_scripts.html#transformer-trained-on-wikitext-2>`_,
and the benchmark definition is given in
:func:`~benchmarking.commons.benchmark_definitions.transformer_wikitext2.transformer_wikitext2_benchmark`.
The script is a little more complex than ``resnet_cifar10``, but has been
modified from open source code in much the same way as in the example above.

In this experiment, we would like to compare different methods w.r.t. their
scalability with the number of workers. To this end, we run a number of HPO
methods with 2, 4, and 8 workers (we show only 2 of the 4 methods we compare,
for simplicity):

.. literalinclude:: ../../../../benchmarking/nursery/tmlr_experiments/tmlr10/baselines.py
   :caption: benchmarking/nursery/tmlr_experiments/tmlr10/baselines.py
   :lines: 13-

As with the example above, ``transformer_wikitext2`` is included in
:func:`~benchmarking.commons.benchmark_definitions.real_benchmark_definitions.real_benchmark_definitions`,
so we can select it from there. Note that ``real_benchmark_definitions`` will be
called with ``sagemaker_backend=True`` in our example here. This means that the
default instance type is ``ml.g4dn.xlarge``, which has one GPU and is cheaper to
use than ``ml.g4dn.12xlarge`` we use above. On the other hand, the SageMaker
backend cannot run multiple workers on one instance, so more training jobs are
required overall.

The scripts ``hpo_main.py`` and ``launch_remote.py`` look exactly the same as
above, except that common code is imported as
:func:`benchmarking.commons.hpo_main_sagemaker.main` and
:func:`benchmarking.commons.launch_remote_sagemaker.launch_remote` now.
While we need to change the number of workers from its default value (which is
4), this is supported by the command line argument ``--n_workers`` which is
supported in the common code already.

Running and Results
-------------------

Let us run this:

.. code-block:: bash

   python benchmarking/nursery/tmlr_experiments/tmlr10/launch_remote.py \
     --experiment_tag demo-offsite-sagemaker --benchmark transformer_wikitext2 \
     --n_workers 2  --warm_pool 1 --scale_max_wallclock_time 1 --num_seeds 1

In order to save time, we only launch one random repetition (seed), for two
methods instead of four, and ask for two workers. Two further options are
relevant:

* ``--warm_pool 1`` activates the recently launched SageMaker warm pooling
  feature, which cuts startup delays by quite a bit
* ``--scale_max_wallclock_time 1`` makes sure that the default for
  ``max_wallclock_time`` is scaled up if the requested number of workers is
  smaller than the default for the benchmark. In our example, we ask for 2
  workers, while the default is 4, so the default ``max_wallclock_time`` is
  doubled

This command is launching two SageMaker training jobs, running the experiment
for each method. Each of these spawns two training jobs at any given time,
running the model training. We will look at the console in a moment.

I ran all of this beforehand, for 5 seeds. If your quota allows for that,
you can run these 20 experiments in parallel, each of which use
``n_workers`` SageMaker training jobs at any given time. If you use warm
pooling, these jobs keep getting reused, so the cost of spinning up an
instance and installing framework dependencies is amortized.

Experiments used ``max_wallclock_time`` of 5 hours for 4 or 8 workers, and
10 hours for 2 workers. We can now look at a comparative plot.

.. |Result transformer_wikitext2| image:: img/tmlr10_transformer_wikitext2.png

+-----------------------------------------------------+
| |Result transformer_wikitext2|                      |
+=====================================================+
| Comparison of methods (left: async; right: sync) on |
| transformer_wikitext2 benchmark.                    |
+-----------------------------------------------------+

* In general, a larger number of workers helps (with the exception for
  ``MOBSTER``, but this may be due to the small number of repetitions)
* However, stepping from random search to model-based HPO helps a lot more
  in this example. ``BO`` (Bayesian optimization) with 4 workers easily
  outperforms ``RS`` (random search) with 8 workers, despite being half as
  expensive to run. In much the same way, ``MOBSTER`` (model-based ASHA)
  with 2 workers strongly outperforms ``ASHA`` with 8 workers, at a fourth
  of the cost

Apart from demonstrating the viability of distributed tuning with the
SageMaker backend, these results show that automated tuning is not solved
by running more parallel hardware, if decisions are not done in an
intelligent way.

Let us now have a look at the *SageMaker Console* (where the warm pooling
feature can also be observed).
