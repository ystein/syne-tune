What Is New in Syne Tune?
=========================

We would like to end this overview with pointing out new contributions to Syne
Tune. If any of them (or anything else) resonates with you, please do not
hesitate to get in touch.

Documentation
-------------

The documentation of Syne Tune is hosted now on ``readthedocs`` and has been much
improved. We provide many examples, a comprehensive FAQ, as well as a growing number
of tutorials. Two of these tutorials are from external contributors:

* `Progressive ASHA <../pasha/pasha.html>`_, contributed by
  `Ondre <https://github.com/ondrejbohdal>`_.
* `Using Syne Tune for Transfer Learning <../transfer_learning/transfer_learning.html>`_,
  contributed by `Sigrid <https://github.com/sighellan>`_.

We have also released a
`developer tutorial <../developer/README.html>`_, which we will continue to
extend in order to encourage contributions to Syne Tune.

Benchmarking
------------

Experimentation with Syne Tune is substantially simplified by the ``benchmarking``
mechanism, as demonstrated here. More details are given in
`this tutorial <../benchmarking/README.html>`_.

**Next steps**:

* Include code for plotting results (as has been used here)

Improved Workflows for Continuous Integration and Testing
---------------------------------------------------------

With Wes, we have an engineer on the team, who hit the ground running, and already
improved a number of mechanisms:

* Running more tests in CI
* Framework for integration tests
* Much improved SIM tracking and sprint planning
* Starting more initiatives to gain internal adoption in AWS and beyond

**Next steps**:

* We are in contact with various teams across Amazon and collect their requirements
  for integrating Syne Tune.

Fast Comparisons with Simulations
---------------------------------

One feature distinguishing Syne Tune from all open source solutions we are aware
of, is that realistic experiments can be simulated on tabulated (or surrogate)
benchmarks, for any method implemented in Syne Tune, and any number of workers.
This means that experiments can be run orders of magnitude faster than real time,
and on cheap instances:

* Allows to rapidly compare against many baselines, using many seeds in order
  to average out random fluctuations
* New ideas of your scientists can be evaluated very rapidly
* Integration and regression tests can be run at very low costs, which allows
  to run many more tests, and catch bugs or performance degredations early

We spent efforts to incorporate a large number of publicly available surrogate
benchmarks.

**Next steps**:

* Integrate additional publicly available surrogate benchmarks
* Suite of integration and regression tests, based on simulation on the existing
  range of surrogate benchmarks

Recent SotA Multi-Fidelity Methods
----------------------------------

Besides ASHA and MOBSTER, Syne Tune also provides impementations of more recent
methods, such as
`HyperTune <tutorials/multifidelity/mf_async_model.html#hyper-tune>`_,
:class:`~syne_tune.optimizer.baselines.ASHABORE`, or
`DyHPO <tutorials/multifidelity/mf_async_model.html#dyhpo>`_. As also seen in
this tutorial, model-based multi-fidelity methods provide good results with
substantially less resources across a wide range of benchmarks. Syne Tune also
provides a range of synchronous multi-fidelity methods.

**Next steps**:

* Currently, AMT offers random search, Bayesian optimization, and a variant of
  ASHA. Our results across the board show that mode-based multi-fidelity methods
  can work much better, in particular when costs are large.

Contributions from the CoSyne Team
----------------------------------

The open source library `Renate <https://github.com/awslabs/Renate>`_ for
continual learning depends on Syne Tune, and team members or interns
regularly contribute to Syne Tune in a major way:

* RUSH (:class:`~syne_tune.optimizer.schedulers.transfer_learning.RUSHScheduler`):
  `Zappella, et al. (2021) <https://arxiv.org/abs/2103.16111>`_. This is
  combining ASHA or MOBSTER with a dependence on data from earlier, related
  HPO experiments. By using well-performing configurations from this data as
  guardrails in the stopping or promotion decisions of ASHA, good configurations
  can often be found with very few evaluations for the current task. There are
  plans to integrate RUSH into AMT.
* `PASHA <tutorials/pasha/pasha.html>`_:
  `Bohdal, et al. (2022) <https://arxiv.org/abs/2207.06940>`_. In ASHA or
  MOBSTER, the maximum number of training epochs needs to be specified up front.
  If a too small value is chosen, all trials may remain undertrained. But
  choosing a large value can result in wasteful early evaluations. Progressive
  ASHA provides a solution, whereby the number of epochs is initially small, but
  is increased in an intelligent fashion. Compared to ASHA, good solutions can be
  found in substantially less wallclock time.

Multi-Objective HPO
-------------------

Substantial efforts were invested in multi-objective methods, whereby we either
face a constrained optimization problem with unknown criterion and constraint,
or a number of objectives with no particular preference. In the latter case, the
goal is not so much optimization, but efficient sampling of the Pareto front,
which represents optimal trade-offs between the different objectives.

Syne Tune contains a number of competitive multi-objective methods, including a
novel method :class:`~syne_tune.optimizer.schedulers.multiobjective.MOASHA` of
our own.

* MO-ASHA has been used to automate fine-tuning of Hugging Face NLP models, with
  a twist (see example in :mod:`benchmarking.nursery.fine_tuning_transformer_glue`).
  Namely. we jointly optimize over optimizer hyperparameters (e.g., learning rate),
  the choice of pre-trained model from the Hugging Face zoo, **and** the AWS
  instance type, while objectives of interest are ``accuracy``, ``cost`` and
  ``latency``. Results have been published in this
  `AWS blog post <https://aws.amazon.com/blogs/machine-learning/hyperparameter-optimization-for-fine-tuning-pre-trained-transformer-models-from-hugging-face/>`_.

**Next steps**:

* Aaron is exploring multi-objective tuning as part of a shared goal to
  integrate *Neural Architecture Search (NAS)* into AMT
* Jacek is refactoring and extending the multi-objective code in Syne Tune in
  order to make it ready for instance type and count tuning as part of the
  *Thanos* service.
