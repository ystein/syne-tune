How Do Methods Differ From Each Other?
======================================

In this section, we would like to give an idea about the properties that make
methods in Syne Tune different from each other. Instead of explaining how each
method works (there are links to such details in our documentation), we will
instead plot learning curves per trial along the common wallclock time, which
will characterize each method by what decisions it makes at which point in time,
given that total resourced and overall experiment time is the same.

These per-trial plots are generated from an experiment on the tabulated
``nas201-ImageNet16-120`` benchmark, which is more difficult than
``resnet_cifar10`` above, and also experiments can be simulated, so that data
is obtained orders of magnitude faster. We use 8 workers for this benchmark, and
a wallclock time of 8 hours. The plots are from a common seed, so that initial random
decisions are the same across all methods.

Random Search and Bayesian Optimization
---------------------------------------

Here is a per-trial plot for random search (RS) and Bayesian optimization (BO),
where the learning curve of each trial has a different color.

.. |fifo_nas201-ImageNet16-120| image:: img/demo_offsite_fifo_nas201-ImageNet16-120.png

+--------------------------------------------------------+
| |fifo_nas201-ImageNet16-120|                           |
+========================================================+
| Learning curves for each trial, for random search (RS) |
| and Bayesian optimization (BO).                        |
+--------------------------------------------------------+

* Both BO and RS train every trial for the full number of epochs (200). Since we use
  8 workers, and full training can require more than 10000 seconds, the initial part
  of the plots are the same (recall they share the same random seed).
* Still, BO has a slight edge and tends to start trials later one which perform
  better, while RS simply starts trials with randomly chosen configurations


Asynchronous Multi-Fidelity: ASHA and MOBSTER
---------------------------------------------

Here is the same plot for two asynchronous multi-fidelity methods. Both:

* pause trials early (at certain rung levels) and only resume trials which outperform
  most others at the same number of epochs
* make decisions whenever a worker becomes available, instead of synchronizing decisions
  until many competing trials reached the same number of epochs

However, ``ASHA`` chooses configurations for new trials at random (like ``RS``), while
``MOBSTER`` uses multi-task Bayesian optimization for such decisions (like ``BO``).

.. |multifid_nas201-ImageNet16-120| image:: img/demo_offsite_multifid_nas201-ImageNet16-120.png

+-------------------------------------------------------+
| |multifid_nas201-ImageNet16-120|                      |
+=======================================================+
| Learning curves for each trial, for ASHA and MOBSTER. |
+-------------------------------------------------------+

* ASHA and MOBSTER train fewer trials for the full number of epochs, while most trials
  are paused early and never resumed. This way, they find significantly better solutions
  in the limited time of the experiment
* MOBSTER manages to start more well-performing trials in the latter half of the
  experiment, and wastes less time on starting trials which do not get far due to a poor
  configuration. The Bayesian optimization mechanism of MOBSTER rapidly learns what not
  to do, while ASHA continues to sample poor configurations at a fixed rate
