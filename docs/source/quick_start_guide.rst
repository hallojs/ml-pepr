Quick-Start Guide
=================
This guide is intended to get you familiar with the library and get things up and running as quickly as possible. Here
we only describe how individual attacks with a single attack configuration are executed. For information about the
attack runner and structured pentesting, see :ref:`attack runner <label-attack-runner>`.

Library Structure
-----------------

The library is kept very simple. You can execute an attack already with two function calls. And to generate an attack
report, you need just one more. For example, for the ``privacy.mia`` attack, it would look as follows (for details about
the function calls, see the documentation of the individual attacks):

1. Create an attack object: ``attack_object = mia.Mia(...)``
2. Run the attack: ``attack_object.run(...)``
3. Create an attack report (optional): ``attack_object.create_attack_report(...)``

That way, also multiple instances of a model can be attacked to ensure that it is not the stochastic learning process
that leads to a privacy-preserving and/or robust model but the chosen hyperparameters. For this purpose, you must pass a
list of target model instances to the constructor of the attack object. To generate balanced training datasets for such
a set of target models, you can use :meth:`pepr.utilities.assign_record_ids_to_target_models`.

.. note:: In the current version, the report generator can only generate reports for attacks on individual target
          models. If the report generator is called on an attack object with multiple target models, it generates an
          attack report for the first target model.

.. note:: To call ``attack_object.create_attack_report(pdf=True)`` you need a full LaTeX installation on your system.

Logging
-------

For logging, we use Python's standard library. For an example of how logging can look like, check out the example
notebooks. For more details, please refer to the `howto <https://docs.python.org/3/howto/logging.html>`_ of the standard
library.

Rapidly Re-Executing Attacks
----------------------------

We want to point out that you can re-execute some attacks without recomputing all the steps of the attack.
For example, the privacy.gmia attack can be re-executed without having to re-train the expensive reference models. This
can be done by calling the ``attack_object.run()`` method again with the ``load_pars`` parameter. For details, please
consult the documentation of the corresponding attack.

.. warning:: Please use this feature only if you know what you are doing. Otherwise, you can easily get unexpected
             results.

Examples
--------

For full and concrete examples checkout our
`example notebooks <https://github.com/hallojs/ml-pepr/tree/master/notebooks>`_:

+----------------------------+------------------------------------------------------------------------------+--------------+
| Notebook                   | Description                                                                  | Google Colab |
+============================+==============================================================================+==============+
| ``mia_tutorial``           | Attack a single target model with mia.                                       | |nb0|_       |
+----------------------------+------------------------------------------------------------------------------+--------------+
| ``direct_gmia_tutorial``   | Attack a single target model with direct-gmia.                               | |nb1|_       |
+----------------------------+------------------------------------------------------------------------------+--------------+
| ``attack_runner_tutorial`` | Organize a pentest.                                                          | Coming soon! |
+----------------------------+------------------------------------------------------------------------------+--------------+

.. |nb0| image:: https://colab.research.google.com/assets/colab-badge.svg
.. _nb0: https://colab.research.google.com/github/hallojs/ml-pepr/blob/master/notebooks/mia_tutorial.ipynb

.. |nb1| image:: https://colab.research.google.com/assets/colab-badge.svg
.. _nb1: https://colab.research.google.com/github/hallojs/ml-pepr/blob/master/notebooks/direct_gmia_tutorial.ipynb